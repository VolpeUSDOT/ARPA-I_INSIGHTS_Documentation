"""Absolute vertical accuracy against USGS 3DEP, without the original control survey.

The georegistration residuals reported in the manuscript were computed against ground
control points that are not part of the release, so they cannot be independently
re-derived or attributed to particular sorties. This script estimates the vertical
difference against an external reference instead, per tile and per sortie, which requires
no access to the original survey.

Reference product
-----------------
The USGS 3DEP **1 m bare-earth DEM** (`s3://prd-tnm/StagedProducts/Elevation/1m/`), not the
AWS-hosted EPT point clouds. The EPT copies are reprojected to EPSG:3857 and declare no
vertical CRS, so their Z would have to be assumed; the staged DEMs are authoritative,
already ground-classified by USGS, carry a declared datum, and are tiled COGs with
overviews so a small window can be read without fetching the whole file.

Datum compatibility, which is what makes the comparison valid at all: INSIGHTS tiles
declare a compound CRS whose vertical component is `NAVD88 height (ftUS)`, and the 3DEP 1 m
DEM is documented as bare-earth elevations "referenced to the North American Vertical Datum
of 1988 (NAVD88)" in metres. Same vertical datum, so only a unit conversion and a
horizontal reprojection are needed. A residual of a few centimetres remains because neither
product states which geoid model realized NAVD88 (GEOID12A/12B versus GEOID18); that floors
the achievable precision but is far below the metre-scale differences of interest.

Selecting the reference tile, which is where a first attempt went wrong
----------------------------------------------------------------------
3DEP tiles are named for their position, in units of 10 km, in the projected CRS of their
project, with the Y field giving the northing of the tile's TOP edge. Two naming
generations coexist:

    USGS_1M_13_x38y442_CO_DRCOG_2020_B20.tif          zone in the name
    USGS_one_meter_x23y437_CO_Central_Western_2016.tif  no zone in the name

Indexing the second form by `(x, y)` alone is unsound, because those coordinates mean
nothing without the CRS they are expressed in. Doing so silently merges Colorado and Utah:
44 of the zone-less keys in these two states are claimed by a project in each, and a
newest-project-first rule then hands a Colorado query a Utah raster. That is not a
hypothetical -- it is what removed `I70D` and `I70E` from an earlier run of this script
entirely, along with 45 of 170 tile comparisons, because the reads against the wrong
raster raised rather than returning bad numbers. `CO_Central_Western_2016` covers both
sorties perfectly well; the lookup was taking the coverage away.

The project name cannot be trusted for this either: `CO_MesaCo_QL2_UTM12_2016` is in fact
EPSG:26913, UTM zone 13. Nor can one CRS be assumed per project, which is the same mistake
one level deeper and cost a second run: 18 of the 61 projects over these two states hold
rasters in more than one projected CRS -- 12 of them declare two zones in their file names,
and the rest are zone-less groups that turn out to straddle a zone boundary, which is why
several rasters spread across such a group are probed rather than one. `CO_NorthwestCO_2020_D20`
spans zone 12 at x66-x76 and zone 13 at x23-x50. Labelling that project zone 12 wholesale made its zone-13 rasters
answer queries near Utah Lake, whose zone-12 easting also falls at x44 -- and because the
two numeric ranges overlap, the read succeeded and returned northwest Colorado terrain
instead of failing. Five I15South tiles came back with differences near -2000 m.

The lesson is that an index built from file names can generate candidates but must never
be trusted for correctness. So the CRS used to sample a raster is the one that raster
declares, read from the open dataset, and every query point is checked against the
raster's own bounds before it is used. A candidate whose 10 km cell matched but which does
not actually cover the query -- whether because the index mislabelled its zone or simply
because a project's edge tiles are clipped to its boundary -- is skipped rather than fatal,
falling through to the next-newest project.

Comparing INSIGHTS against a BARE-EARTH reference
-------------------------------------------------
This is the other part that needs care, and a first attempt also got it wrong. Taking a low
percentile of elevation within a cell and requiring the within-cell spread to be small does
select flat surfaces -- but a flat roof is flat, so buildings passed the filter and were
differenced against the bare ground beneath them, producing differences of +12 m and worse.

The fix is a morphological opening, the same principle as a progressive morphological ground
filter. The per-cell low percentile is eroded and then dilated with a window wider than a
building, which removes objects that sit above their surroundings while preserving terrain
including sloped terrain. A cell is used only if its value is close to that opened surface,
i.e. only if it is at the bare-earth level. Crucially this uses no information from the DEM,
so it cannot bias the comparison toward agreement.

What is reported per tile
-------------------------
The median difference is the tile's vertical bias. Beyond that, a plane is fitted to the
difference field across the tile, because a bias and a tilt are different defects and the
median alone cannot distinguish them: `grad_mm_per_m` is the gradient of that plane and
`nmad_detrended_m` the scatter remaining after it is removed. Comparing `nmad_m` against
`nmad_detrended_m` says whether the difference over a tile is flat with noise or
systematically sloped. `ramp_fit.py` uses the per-tile medians to ask the same question at
the scale of a whole sortie.

Limits of what this measures
----------------------------
* Vertical only. On flat ground a horizontal misregistration produces almost no vertical
  difference, so this is insensitive to horizontal error -- the same blind spot as the
  sortie-to-sortie comparison. Horizontal accuracy needs feature matching against imagery.
* The reference has its own error, typically <= 10 cm RMSEz for the QL1/QL2 projects used.
* Real change between the reference epoch and June 2025 is a confound. Restricting to flat
  bare ground and reporting robust statistics limits it, but a sortie compared against a
  2016 reference is weaker evidence than one compared against 2023. `gap_yr` reports it.
"""

import argparse
import io
import json
import os
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")

import boto3
import laspy
import numpy as np
import pandas as pd
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.ndimage import grey_dilation, grey_erosion

M_PER_FTUS = 0.3048006096012192
CELL_M = 2.0                 # comparison grid
MIN_PTS_PER_CELL = 30
MAX_SPREAD_M = 0.5           # within-cell flatness, measured on the GROUND portion only
OPEN_CELLS = 21              # morphological window, 42 m: still wider than a building
BARE_TOL_M = 0.5             # base tolerance above the opened surface, before slope
MIN_CELLS = 100              # per tile, to accept an estimate
MAX_CANDIDATES = 8           # reference projects to try before giving up on a tile

# The release lies entirely within Colorado and Utah, so every 1 m project over those two
# states is a candidate and the list is discovered rather than hardcoded. Discovery matters:
# a hand-maintained list silently omits projects, and CO_ArapahoRooseveltPikeNF_D23 (2023)
# and UT_WestEast_B22 (2022) are both newer than anything an earlier version of this script
# knew about.
STATES = ("CO_", "UT_")

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=64))


def project_year(name):
    """Collection or delivery year from a project name, e.g. ..._2020_B20 or ..._D23."""
    m = re.search(r"((?:19|20)\d{2})", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_[A-Z](\d{2})$", name)
    if m:
        return 2000 + int(m.group(1))
    return 0


def list_projects():
    pag = s3.get_paginator("list_objects_v2")
    out = []
    for pg in pag.paginate(Bucket="prd-tnm",
                           Prefix="StagedProducts/Elevation/1m/Projects/", Delimiter="/"):
        for p in pg.get("CommonPrefixes", []):
            n = p["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if n.startswith(STATES):
                out.append(n)
    return sorted(out)


def project_cells(name):
    """{zone_or_None: {(x_10km, y_10km_top): key}} for one project.

    Grouped by the zone in the file name, because a single project can hold rasters in
    more than one UTM zone and a project-wide CRS would mislabel all but one of them.
    """
    pag = s3.get_paginator("list_objects_v2")
    groups = {}
    for pg in pag.paginate(Bucket="prd-tnm",
                           Prefix=f"StagedProducts/Elevation/1m/Projects/{name}/TIFF/"):
        for o in pg.get("Contents", []):
            k = o["Key"]
            if not k.endswith(".tif"):
                continue
            m = re.search(r"_(\d{2})_x(\d+)y(\d+)_", k)
            if m:
                zone, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                m = re.search(r"_x(\d+)y(\d+)_", k)
                if not m:
                    continue
                zone, x, y = None, int(m.group(1)), int(m.group(2))
            groups.setdefault(zone, {})[(x, y)] = k
    return groups


def probe_epsgs(keys):
    """Distinct projected CRSs among a sample of a group's rasters.

    Used only for the older `USGS_one_meter_*` products, whose file names carry no zone.
    Several rasters spread across the group are probed rather than one, so that a
    zone-straddling group is registered under every CRS it actually uses instead of being
    silently collapsed onto the first one seen.
    """
    out = []
    for k in keys:
        try:
            with rasterio.open(f"https://prd-tnm.s3.amazonaws.com/{k}") as ds:
                e = ds.crs.to_epsg()
            if e and int(e) not in out:
                out.append(int(e))
        except rasterio.errors.RasterioError:
            pass
    return out


def build_index(cache, workers=12, refresh=False):
    """[(year, project, epsg, {cell: key}), ...] newest first, one entry per (project, CRS).

    The epsg is a candidate-generation hint only. It comes from the zone in the file name
    where there is one, and otherwise from probing a sample of the group's rasters. Either
    way the sampling in `read_reference` re-derives the CRS from the raster itself, so an
    entry that is mislabelled here costs a wasted candidate rather than a wrong answer.
    Cached; the listing and the probes together take a couple of minutes.
    """
    if os.path.exists(cache) and not refresh:
        raw = json.load(open(cache))
        idx = [(e["year"], e["project"], e["epsg"],
                {tuple(map(int, k.split(","))): v for k, v in e["cells"].items()})
               for e in raw]
        print(f"  reusing {cache}: {len(idx)} (project, CRS) groups, "
              f"{sum(len(c) for *_, c in idx)} DEM tiles", file=sys.stderr)
        return idx

    names = list_projects()
    print(f"  {len(names)} 1 m projects over {'/'.join(s_.strip('_') for s_ in STATES)}",
          file=sys.stderr)
    with ThreadPoolExecutor(workers) as ex:
        grouped = dict(zip(names, ex.map(project_cells, names)))

    # zone-less groups need probing; spread the probes across the group's x range so that
    # a group straddling two zones is caught rather than assumed away
    to_probe = {}
    for name, groups in grouped.items():
        cells = groups.get(None)
        if cells:
            ks = sorted(cells, key=lambda c: c[0])
            picks = {ks[0], ks[len(ks) // 3], ks[2 * len(ks) // 3], ks[-1]}
            to_probe[name] = [cells[k] for k in picks]
    with ThreadPoolExecutor(workers) as ex:
        probed = dict(zip(to_probe, ex.map(probe_epsgs, to_probe.values())))

    idx = []
    for name, groups in grouped.items():
        year = project_year(name)
        for zone, cells in groups.items():
            if zone is not None:
                idx.append((year, name, 26900 + zone, cells))
            else:
                for e in probed.get(name, []):
                    idx.append((year, name, e, cells))
    idx.sort(key=lambda t: t[0], reverse=True)
    multi = sorted({n for _, n, _, _ in idx
                    if sum(1 for _, m, _, _ in idx if m == n) > 1})
    for y, n, e, c in idx:
        print(f"    {n:36s} {y}  EPSG:{e}  {len(c):6d} tiles", file=sys.stderr)
    if multi:
        print(f"  {len(multi)} projects hold rasters in more than one CRS: "
              f"{', '.join(multi)}", file=sys.stderr)
    json.dump([{"year": y, "project": n, "epsg": e,
                "cells": {f"{a},{b}": k for (a, b), k in c.items()}} for y, n, e, c in idx],
              open(cache, "w"))
    return idx


class Reference:
    """Newest-first candidate reference tiles for a query point.

    Keyed on a per-(project, CRS) group rather than per project, so that a project holding
    rasters in two UTM zones cannot answer a query in the wrong one.
    """

    def __init__(self, index):
        self.index = index
        self._tr = {}

    def _to(self, epsg):
        if epsg not in self._tr:
            self._tr[epsg] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}",
                                                  always_xy=True)
        return self._tr[epsg]

    def candidates(self, lon, lat, limit=MAX_CANDIDATES):
        out, seen = [], set()
        for year, proj, epsg, cells in self.index:
            x, y = self._to(epsg).transform(lon, lat)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            # the Y field is the northing of the tile's top edge, hence the +1
            key = cells.get((int(x // 10000), int(y // 10000) + 1))
            if key is not None and key not in seen:
                seen.add(key)
                out.append((key, proj, year))
                if len(out) == limit:
                    break
        return out


def read_reference(demkey, src, px, py):
    """Sample one 3DEP raster at the query points, in the CRS the raster itself declares.

    Returns (refz, keep, epsg, ux, uy) for the points that fall inside the raster and carry
    data, or a string naming why this candidate was rejected. Nothing here relies on the
    index being right about the raster's CRS: that is read from the open dataset, and the
    points are checked against the raster's own bounds. Retries before giving up, because
    treating a transient network failure as "this project does not cover the tile" would
    quietly substitute an older reference and inflate the apparent temporal gap.
    """
    url = f"https://prd-tnm.s3.amazonaws.com/{demkey}"
    err = "read-failed"
    for _ in range(3):
        try:
            with rasterio.open(url) as ds:
                epsg = ds.crs.to_epsg()
                tr = Transformer.from_crs(src, ds.crs.to_wkt(), always_xy=True)
                ux, uy = tr.transform(px, py)
                b = ds.bounds
                inb = (np.isfinite(ux) & np.isfinite(uy) &
                       (ux >= b.left) & (ux <= b.right) &
                       (uy >= b.bottom) & (uy <= b.top))
                if int(inb.sum()) < MIN_CELLS:
                    return f"outside-raster({int(inb.sum())})"
                w = from_bounds(max(ux[inb].min() - 5, b.left),
                                max(uy[inb].min() - 5, b.bottom),
                                min(ux[inb].max() + 5, b.right),
                                min(uy[inb].max() + 5, b.top), ds.transform)
                # Snap to whole pixels before reading. from_bounds returns a float window;
                # ds.read rounds it to decide the array while window_transform does not,
                # so leaving it unrounded lets the array origin and the transform used to
                # index into it disagree by up to a pixel, i.e. a metre of horizontal
                # sampling error.
                w = w.round_offsets(op="floor").round_lengths(op="ceil")
                arr = ds.read(1, window=w)
                tf = ds.window_transform(w)
                nod = ds.nodata
            col = np.full(ux.shape, -1, np.int64)
            rw = np.full(uy.shape, -1, np.int64)
            col[inb] = ((ux[inb] - tf.c) / tf.a).astype(np.int64)
            rw[inb] = ((uy[inb] - tf.f) / tf.e).astype(np.int64)
            keep = inb & (rw >= 0) & (rw < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
            if int(keep.sum()) < MIN_CELLS:
                return f"window-miss({int(keep.sum())})"
            refz = arr[rw[keep], col[keep]].astype(float)
            if nod is not None:
                ok = refz != nod
            else:
                ok = np.isfinite(refz)
            if int(ok.sum()) < MIN_CELLS:
                return f"nodata({int(ok.sum())})"
            idxk = np.nonzero(keep)[0][ok]
            fk = np.zeros(ux.shape, bool)
            fk[idxk] = True
            return refz[ok], fk, epsg, ux, uy
        except rasterio.errors.RasterioError as e:
            err = f"read-failed({repr(e)[:40]})"
    return err


def ground_cells(x, y, z, minx, miny):
    """Per-cell low percentile, then keep only cells at the bare-earth level.

    Returns (col_idx, row_idx, elevation_ft, n_before_bare_filter, n_after).
    """
    cell = CELL_M / M_PER_FTUS
    ix = ((x - minx) / cell).astype(np.int64)
    iy = ((y - miny) / cell).astype(np.int64)
    nx, ny = ix.max() + 1, iy.max() + 1
    flat = ix * ny + iy
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], z[order]
    bounds = np.searchsorted(fs, np.arange(nx * ny + 1))
    cnt = np.diff(bounds)
    p10 = np.full(nx * ny, np.nan)
    spread = np.full(nx * ny, np.nan)
    for c in np.nonzero(cnt >= MIN_PTS_PER_CELL)[0]:
        seg = zs[bounds[c]:bounds[c + 1]]
        # Flatness must be judged on the GROUND portion of the cell, not the whole
        # distribution. A p10-to-p90 spread is destroyed by anything above the surface:
        # canopy, buildings, and in this release the high-noise and banded artifact
        # layers, which rejected almost every tile in the worst-affected sorties.
        a, b = np.percentile(seg, [10, 50])
        p10[c], spread[c] = a, b - a
    g = p10.reshape(nx, ny)
    flatm = (spread.reshape(nx, ny) * M_PER_FTUS) < MAX_SPREAD_M
    have = np.isfinite(g)
    n_flat = int((have & flatm).sum())

    # morphological opening: erode then dilate with a window wider than a building.
    # Objects sitting above their surroundings are removed; terrain, including slopes,
    # survives. Uses no DEM information, so it cannot bias the comparison.
    filled = np.where(have, g, np.nanmax(g) if have.any() else 0.0)
    opened = grey_dilation(grey_erosion(filled, size=(OPEN_CELLS, OPEN_CELLS)),
                           size=(OPEN_CELLS, OPEN_CELLS))
    # A fixed tolerance rejects sloped ground, because an opening over a wide window
    # necessarily sits below a slope. Allow for the drop the local gradient implies
    # across half the window, so the test stays a building test rather than a slope test.
    gy, gx = np.gradient(opened * M_PER_FTUS, CELL_M)
    slope_drop = np.hypot(gx, gy) * (OPEN_CELLS * CELL_M / 2.0)
    bare = (g - opened) * M_PER_FTUS <= (BARE_TOL_M + slope_drop)

    sel = have & flatm & bare
    ci, ri = np.nonzero(sel)
    return ci, ri, g[ci, ri], n_flat, int(sel.sum())


def plane_fit(px, py, d, trim=3.0, iters=4):
    """Robustly fit d = a + b*x + c*y over the tile.

    Returns (gradient mm/m, detrended NMAD, cells retained).

    A tile-wide bias and a tile-wide tilt are different defects, and the median cannot tell
    them apart. The fit must be robust to be usable as evidence for a tilt: plain least
    squares is dragged by the tail this difference field carries even after the bare-earth
    restriction -- steep terrain edges, unremoved vegetation, real change, and DEM artifacts
    in the mountain corridors. On one I70E tile with a median of -1.14 m but a 5th percentile
    of -4.38 m, least squares returned a 15.6 mm/m gradient whose removal made the robust
    scatter worse, 0.26 m to 0.70 m, which is the signature of a fit chasing outliers rather
    than a real slope. Iterative trimming at 3 NMAD removes that.
    """
    A = np.c_[np.ones(d.size), px - px.mean(), py - py.mean()]
    keep = np.ones(d.size, bool)
    coef = np.zeros(3)
    for _ in range(iters):
        if keep.sum() < 3 * A.shape[1]:
            break
        coef, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        res = d - A @ coef
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        if not s > 0:
            break
        nk = np.abs(res - np.median(res[keep])) <= trim * s
        if nk.sum() == keep.sum() and (nk == keep).all():
            break
        keep = nk
    res = d[keep] - A[keep] @ coef
    return (float(np.hypot(coef[1], coef[2]) * 1000.0),
            float(1.4826 * np.median(np.abs(res - np.median(res)))),
            int(keep.sum()))


def compare(row, ref):
    key = row["s3_path"].replace("s3://arpa-i-insights/", "")
    buf = io.BytesIO()
    s3.download_fileobj("arpa-i-insights", key, buf)
    buf.seek(0)
    with laspy.open(buf) as f:
        las = f.read()
    x = np.array(las.x, float); y = np.array(las.y, float); z = np.array(las.z, float)
    del las; buf.close()

    ci, ri, gz_ft, n_flat, n_bare = ground_cells(x, y, z, row["minx"], row["miny"])
    base = dict(id=row["id"], sortie=row["sortie"], n_flat=n_flat, n_bare=n_bare)
    if n_bare < MIN_CELLS:
        return dict(base, ok=False, note="too few bare-earth cells")

    cell = CELL_M / M_PER_FTUS
    px = row["minx"] + (ci + 0.5) * cell
    py = row["miny"] + (ri + 0.5) * cell
    src = f"EPSG:{int(row['proj:epsg'])}"
    lon, lat = Transformer.from_crs(src, "EPSG:4326", always_xy=True).transform(
        float(np.mean(px)), float(np.mean(py)))

    cands = ref.candidates(lon, lat)
    if not cands:
        return dict(base, ok=False, note="no 3DEP tile covers this tile")

    tried = []
    for demkey, proj, year in cands:
        got = read_reference(demkey, src, px, py)
        if isinstance(got, str):
            # the 10 km cell matched but this raster does not usably cover the query, so
            # fall through to the next-newest project rather than losing the tile
            tried.append(f"{proj}:{got}")
            continue
        refz, keep, epsg, ux, uy = got

        d = np.asarray(gz_ft[keep] * M_PER_FTUS - refz, float)
        med = float(np.median(d))
        grad, nmad_dt, n_fit = plane_fit(ux[keep], uy[keep], d)
        return dict(base, ok=True, dem_project=proj, dem_year=year, dem_epsg=epsg,
                    n=int(d.size), dz_m=round(med, 4),
                    nmad_m=round(float(1.4826 * np.median(np.abs(d - med))), 4),
                    grad_mm_per_m=round(grad, 3), nmad_detrended_m=round(nmad_dt, 4),
                    n_plane_fit=n_fit,
                    p5=round(float(np.percentile(d, 5)), 3),
                    p95=round(float(np.percentile(d, 95)), 3),
                    max_abs=round(float(np.abs(d).max()), 3),
                    cx=round(float(np.mean(px)), 2), cy=round(float(np.mean(py)), 2),
                    skipped=";".join(tried) or None)
    return dict(base, ok=False, note="no candidate reference covered this tile",
                skipped=";".join(tried))


def interior_sample(index_path, n_per_sortie, seed):
    d = pd.read_parquet(index_path, columns=["id", "sortie", "proj:bbox", "proj:epsg",
                                             "s3_path"])
    bb = np.array([json.loads(x) for x in d["proj:bbox"]], dtype=float)
    d["minx"], d["miny"], d["maxx"], d["maxy"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    xy = d["id"].str.extract(r"_X(\d+)_Y(\d+)")
    d["gx"], d["gy"] = xy[0].astype(int), xy[1].astype(int)
    keep = []
    for s, g in d.groupby("sortie", sort=False):
        present = set(zip(g["gx"], g["gy"]))
        inter = np.array([
            all((x + a, y + b) in present for a in (-1, 0, 1) for b in (-1, 0, 1)
                if (a, b) != (0, 0)) for x, y in zip(g["gx"], g["gy"])])
        gi = g[inter]
        rng = np.random.default_rng(zlib.crc32(s.encode()) + seed)
        keep.append(gi.iloc[rng.choice(len(gi), min(n_per_sortie, len(gi)), replace=False)])
    return pd.concat(keep, ignore_index=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(repo, "data", "lidar", "v1", "stac",
                                                    "index", "items.parquet"))
    ap.add_argument("--n", type=int, default=30, help="interior tiles per sortie")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sortie", action="append", help="restrict to these sorties")
    ap.add_argument("--refresh-dem-index", action="store_true",
                    help="re-list 3DEP and re-probe each project's CRS")
    ap.add_argument("--out", default=os.path.join(here, "out", "compare_3dep.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    print("indexing 3DEP 1 m projects", file=sys.stderr)
    ref = Reference(build_index(os.path.join(os.path.dirname(a.out), "dem_index.json"),
                                refresh=a.refresh_dem_index))

    samp = interior_sample(a.index, a.n, a.seed)
    if a.sortie:
        samp = samp[samp["sortie"].isin(a.sortie)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l).get("id") for l in open(a.out)}
        print(f"resuming, {len(done)} tiles already done", file=sys.stderr)
    todo = samp[~samp["id"].isin(done)]
    print(f"comparing {len(todo)} tiles with {a.workers} workers", file=sys.stderr)

    errs = []
    with open(a.out, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(compare, r, ref): r["id"] for _, r in todo.iterrows()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                fh.write(json.dumps(f.result()) + "\n"); fh.flush()
            except Exception as e:
                errs.append((futs[f], repr(e)[:200]))
            if i % 25 == 0:
                print(f"  {i}/{len(futs)}  ({len(errs)} errors)", file=sys.stderr,
                      flush=True)

    allrows = [json.loads(l) for l in open(a.out)]
    d = pd.DataFrame(allrows).drop_duplicates(subset="id", keep="last")
    good = d[d.ok == True] if "ok" in d else d
    pd.set_option("display.width", 220)

    print(f"\nusable tiles: {len(good)} of {len(d)} attempted; errors this pass: {len(errs)}")
    # Print every error, not the first three. An earlier version printed three of 45 and
    # the two sorties that dropped out of the results entirely went unnoticed.
    for tid, msg in errs:
        print(f"  err {tid} {msg}")
    if "note" in d and d.ok.eq(False).any():
        print("\nrejections:")
        print(d[d.ok == False].groupby("note").size().to_string())

    print("\n" + "=" * 104)
    print("ABSOLUTE VERTICAL DIFFERENCE, INSIGHTS MINUS 3DEP 1 m BARE-EARTH DEM")
    print("=" * 104)
    agg = good.groupby("sortie").agg(
        tiles=("dz_m", "size"), cells=("n", "sum"),
        dz_median=("dz_m", "median"),
        dz_min=("dz_m", "min"), dz_max=("dz_m", "max"),
        spread=("dz_m", lambda v: float(v.max() - v.min())),
        within_tile_nmad=("nmad_m", "median"),
        detrended_nmad=("nmad_detrended_m", "median"),
        within_tile_grad=("grad_mm_per_m", "median"),
        ref_year=("dem_year", lambda v: int(v.median())),
    )
    agg["gap_yr"] = 2025 - agg["ref_year"]
    print(agg.round(3).to_string())
    print("\n  dz_median is the per-sortie vertical bias against the reference.")
    print("  within_tile_nmad is the scatter inside a tile: the measurement's precision.")
    print("  detrended_nmad is that scatter after a tile-wide plane is removed, and")
    print("  within_tile_grad the gradient of that plane, in mm per m: together they say")
    print("  whether the difference over a tile is flat with noise or systematically sloped.")
    print("  spread across tiles indicates whether the bias is uniform over the sortie;")
    print("  ramp_fit.py tests whether that variation is a ramp along the flight direction.")
    print("  gap_yr is years between the reference epoch and the June 2025 collection;")
    print("  a large gap means real change contributes to the difference.")
    agg.to_csv(a.out.replace(".jsonl", "_by_sortie.csv"))

    miss = sorted(set(samp["sortie"]) - set(good["sortie"]))
    if miss:
        print(f"\nWARNING: no usable tile for {', '.join(miss)}")
    print(f"\nwrote {a.out} and {a.out.replace('.jsonl', '_by_sortie.csv')}")


if __name__ == "__main__":
    main()
