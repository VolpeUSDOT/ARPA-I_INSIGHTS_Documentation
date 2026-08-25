"""Horizontal registration against USGS 3DEP, without the original control survey.

The horizontal residuals reported in the manuscript come from ground control points that
are not part of the release, so nothing published lets a user check them. Every other
analysis in this directory is deliberately blind to horizontal error: they all restrict
themselves to flat ground, where a horizontal misregistration produces almost no vertical
difference. This script measures the quantity those analyses cannot see, and it does so by
inverting their restriction -- it needs terrain with *slope*, and it is the flat sorties
that it cannot say much about.

The method
----------
Where a surface is displaced horizontally, the displacement shows up in elevation in
proportion to the terrain gradient. Write the true surface as H. If INSIGHTS reports, at
grid position (x, y), the terrain that actually lies at (x + dx, y + dy), then to first
order

    dh = h_insights - h_reference = dz + dx * dH/dx + dy * dH/dy

so regressing the elevation difference on the two components of the reference gradient
recovers the horizontal shift as the regression coefficients. This is the derivative form
of Nuth and Kaeaeb (2011) co-registration; the cosine-of-aspect form in that paper is the
same relation in polar coordinates, and the linear form used here needs no iteration and
has an exact covariance.

Sign convention: `(dx, dy)` is the translation that must be **applied to INSIGHTS** to bring
it onto the reference. A positive `dx` means INSIGHTS features sit that far west of where
the reference puts them.

The trap that makes a naive version of this wrong
-------------------------------------------------
The per-cell ground proxy used throughout this directory is a low percentile of elevation
within the cell. On flat ground that is unbiased and harmless. On a slope it is not: the
10th percentile of a cell straddling a gradient sits toward the cell's downhill edge, low
by roughly 0.4 * |grad H| * cell, and that error grows with slope and points downhill.

That is the same shape as the signal being fitted. A 50% slope biases the ground estimate
by about 0.4 m, while the elevation difference a 1 m horizontal shift produces on that slope
is 0.5 m -- comparable. Left alone it would be read as a horizontal shift that is really an
artefact of the estimator.

The fix is to give the bias its own term. Slope magnitude is carried as a fourth regressor,

    dh = dz + dx * dH/dx + dy * dH/dy + beta * |grad H|

which absorbs it, along with any other slope-dependent difference between the two products.
It costs almost nothing in variance because |grad H| is even in aspect while the two gradient
components are odd, so over a tile with mixed aspects they are nearly orthogonal. The fitted
beta is reported: it should land near -0.4 * cell, and if it does the mechanism above is
confirmed rather than assumed. `--no-slope-term` drops it, for comparison.

Conditioning, which decides whether an answer means anything
-----------------------------------------------------------
On a *uniform* slope the two gradient components are constant across the tile, so they are
collinear with the intercept and dz cannot be separated from the shift. What identifies a
shift is not slope but *variation* in slope and aspect: a valley, a ridge, undulating
terrain. Consequently

* every estimate is reported with a standard error -- but *not* the one the fit hands back.
  The formal error treats each 2 m cell as an independent observation, and the residuals are
  correlated over tens of metres, so it is optimistic by an order of magnitude and yields
  absurdities like a shift of 1.3 m "+/- 0.01 m". The reported error is the formal one
  inflated by the scatter of the per-tile estimates about the pooled value, and the inflation
  factor is printed so the reader can see how badly independence fails;
* the headline per-sortie estimate is a **pooled** fit over all of that sortie's tiles at
  once, carrying **one vertical offset per tile** as a nuisance parameter and sharing only
  (dx, dy, beta). Per-tile offsets are essential rather than tidy: the vertical bias varies
  along these sorties by metres, as `ramp_fit.py` shows, and a single offset would push that
  variation into the shift estimate. Pooling also buys aspect diversity, because tiles that
  are each poorly conditioned alone can be well conditioned together if their terrain faces
  different ways.

Because the pooled fit only ever needs cross-products, each tile stores the small centred
cross-product matrix of its own cells rather than the cells themselves, and the per-sortie
solve is exact from those.

Validating the estimator instead of trusting it
-----------------------------------------------
`--synthetic-shift DX,DY` translates the INSIGHTS points by a known amount before fitting,
so the estimator can be checked against ground truth on real data. Translating by T leaves a
correction of (base - T), so recovery is measured against a baseline run via `--baseline`,
not against zero: the tiles carry a real shift already. Use `--cache-dir` so the repeated
runs this needs do not re-download anything.

What the test showed, and it is the right way to read the numbers this script produces: the
recovery is close to unbiased at metre scale -- an applied (+1.50, -0.75) m came back as
(+1.51, -0.79) m -- but the error does not shrink in proportion for smaller shifts, so there
is an accuracy floor of roughly 0.2 to 0.3 m per sortie rather than a multiplicative
attenuation. A measured shift of a metre or more is real; a measured shift of 0.3 m is not
distinguishable from zero, whatever the fit's own standard error says.

Limits
------
* Needs terrain. Over the flat urban sorties the standard errors will say so; do not read a
  small shift with a large standard error as evidence of good registration.
* First order in the shift, so it assumes the displacement is small next to the scale over
  which the gradient varies. At the metre level against a 6 m smoothed gradient, it is.
* The reference has its own horizontal error, roughly a metre for the older projects, and
  that floors what can be resolved.
* Real change between the reference epoch and June 2025 contributes, as in `compare_3dep.py`.
"""

import argparse
import io
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.ndimage import grey_dilation, grey_erosion, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_3dep as C3

M_PER_FTUS = C3.M_PER_FTUS
CELL_M = 2.0                 # comparison grid
MIN_PTS_PER_CELL = 20
BASE_SPREAD_M = 0.5          # within-cell spread allowed before the slope allowance
OPEN_CELLS = 21              # morphological window, 42 m: wider than a building
BARE_TOL_M = 0.5
SMOOTH_CELLS = 3             # 6 m smoothing of the reference before differentiating
MIN_FIT_CELLS = 500          # per tile
MIN_RELIEF_M = 15.0          # tile selection: enough terrain to carry information
MAX_RELIEF_M = 200.0         # and not so much that noise dominates the index statistic


def ground_grid(x, y, z, minx, miny, grad_mag):
    """Per-cell ground proxy on the tile's grid, with a slope-aware spread test.

    Returns (p10_ft, usable_mask). Unlike `compare_3dep.ground_cells` this must NOT reject
    sloped terrain -- slope is the signal here -- so the within-cell spread allowance grows
    with the local gradient instead of being a fixed 0.5 m.
    """
    cell = CELL_M / M_PER_FTUS
    ix = ((x - minx) / cell).astype(np.int64)
    iy = ((y - miny) / cell).astype(np.int64)
    nx, ny = grad_mag.shape
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[ok], iy[ok], z[ok]
    flat = ix * ny + iy
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], z[order]
    bounds = np.searchsorted(fs, np.arange(nx * ny + 1))
    cnt = np.diff(bounds)
    p10 = np.full(nx * ny, np.nan)
    spread = np.full(nx * ny, np.nan)
    for c in np.nonzero(cnt >= MIN_PTS_PER_CELL)[0]:
        seg = zs[bounds[c]:bounds[c + 1]]
        a, b = np.percentile(seg, [10, 50])
        p10[c], spread[c] = a, b - a
    g = p10.reshape(nx, ny)
    spread = spread.reshape(nx, ny) * M_PER_FTUS
    have = np.isfinite(g)

    # what a plane at the local gradient would itself span across one cell
    allow = BASE_SPREAD_M + grad_mag * CELL_M
    clean = have & (spread < allow)

    # morphological opening, as in compare_3dep: remove what stands above its surroundings
    filled = np.where(have, g, np.nanmax(g) if have.any() else 0.0)
    opened = grey_dilation(grey_erosion(filled, size=(OPEN_CELLS, OPEN_CELLS)),
                           size=(OPEN_CELLS, OPEN_CELLS))
    gy, gx = np.gradient(opened * M_PER_FTUS, CELL_M)
    slope_drop = np.hypot(gx, gy) * (OPEN_CELLS * CELL_M / 2.0)
    bare = (g - opened) * M_PER_FTUS <= (BARE_TOL_M + slope_drop)
    return g, clean & bare


def sample_dem_bilinear(demkey, src, ux_fn):
    """Bilinearly sample one 3DEP raster on the tile grid, in the raster's own CRS.

    Bilinear rather than nearest because a nearest-neighbour lookup quantises the reference
    to its 1 m posts, and that quantisation is a large fraction of the sub-metre shift being
    estimated. Returns (values_m, valid_mask, epsg) or a string naming the rejection, with
    the same "trust the raster, not the index" discipline as `compare_3dep.read_reference`.
    """
    url = f"https://prd-tnm.s3.amazonaws.com/{demkey}"
    err = "read-failed"
    for _ in range(3):
        try:
            with rasterio.open(url) as ds:
                epsg = ds.crs.to_epsg()
                ux, uy = ux_fn(ds.crs.to_wkt())
                b = ds.bounds
                inb = (np.isfinite(ux) & np.isfinite(uy) &
                       (ux >= b.left + 2) & (ux <= b.right - 2) &
                       (uy >= b.bottom + 2) & (uy <= b.top - 2))
                if int(inb.sum()) < MIN_FIT_CELLS:
                    return f"outside-raster({int(inb.sum())})"
                w = from_bounds(max(ux[inb].min() - 5, b.left),
                                max(uy[inb].min() - 5, b.bottom),
                                min(ux[inb].max() + 5, b.right),
                                min(uy[inb].max() + 5, b.top), ds.transform)
                w = w.round_offsets(op="floor").round_lengths(op="ceil")
                arr = ds.read(1, window=w).astype(float)
                tf = ds.window_transform(w)
                nod = ds.nodata
            if nod is not None:
                arr[arr == nod] = np.nan
            # pixel-centre coordinates
            cc = (ux - tf.c) / tf.a - 0.5
            rr = (uy - tf.f) / tf.e - 0.5
            c0 = np.floor(cc); r0 = np.floor(rr)
            fc = cc - c0; fr = rr - r0
            c0 = c0.astype(np.int64); r0 = r0.astype(np.int64)
            good = (inb & (c0 >= 0) & (c0 + 1 < arr.shape[1]) &
                    (r0 >= 0) & (r0 + 1 < arr.shape[0]))
            out = np.full(ux.shape, np.nan)
            if int(good.sum()) < MIN_FIT_CELLS:
                return f"window-miss({int(good.sum())})"
            ci, ri, a, b_ = c0[good], r0[good], fc[good], fr[good]
            v = ((1 - a) * (1 - b_) * arr[ri, ci] + a * (1 - b_) * arr[ri, ci + 1] +
                 (1 - a) * b_ * arr[ri + 1, ci] + a * b_ * arr[ri + 1, ci + 1])
            out[good] = v
            valid = np.isfinite(out)
            if int(valid.sum()) < MIN_FIT_CELLS:
                return f"nodata({int(valid.sum())})"
            return out, valid, epsg
        except rasterio.errors.RasterioError as e:
            err = f"read-failed({repr(e)[:40]})"
    return err


def robust_fit(A, d, trim=3.0, iters=4):
    """Trimmed least squares. Returns (coef, cov, n_kept, sigma) or None."""
    keep = np.ones(d.size, bool)
    coef = None
    for _ in range(iters):
        if keep.sum() < 5 * A.shape[1]:
            return None
        coef, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        res = d - A @ coef
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        if not s > 0:
            break
        nk = np.abs(res - np.median(res[keep])) <= trim * s
        if (nk == keep).all():
            break
        keep = nk
    res = d[keep] - A[keep] @ coef
    dof = keep.sum() - A.shape[1]
    if dof <= 0:
        return None
    sigma2 = float((res ** 2).sum()) / dof
    try:
        cov = np.linalg.inv(A[keep].T @ A[keep]) * sigma2
    except np.linalg.LinAlgError:
        return None
    return coef, cov, int(keep.sum()), float(np.sqrt(sigma2))


def load_points(row, cache_dir=None):
    """Tile coordinates, optionally via an on-disk cache.

    The cache exists for validation: recovering a known synthetic shift means running the
    estimator repeatedly over the same tiles, and re-downloading them each time is the
    dominant cost by a wide margin.
    """
    import laspy
    key = row["s3_path"].replace("s3://arpa-i-insights/", "")
    path = os.path.join(cache_dir, row["id"] + ".copc.laz") if cache_dir else None
    if path and os.path.exists(path):
        with laspy.open(path) as f:
            las = f.read()
    else:
        buf = io.BytesIO()
        C3.s3.download_fileobj("arpa-i-insights", key, buf)
        if path:
            with open(path, "wb") as fh:
                fh.write(buf.getbuffer())
        buf.seek(0)
        with laspy.open(buf) as f:
            las = f.read()
        buf.close()
    return (np.array(las.x, float), np.array(las.y, float), np.array(las.z, float))


def process(row, ref, synthetic, use_slope_term, cache_dir=None, null_test=False):
    x, y, z = load_points(row, cache_dir)

    if synthetic is not None:
        # translate the point cloud by a known amount, in metres, to test recovery
        x = x + synthetic[0] / M_PER_FTUS
        y = y + synthetic[1] / M_PER_FTUS

    base = dict(id=row["id"], sortie=row["sortie"])
    cell = CELL_M / M_PER_FTUS
    nx = int(np.ceil((row["maxx"] - row["minx"]) / cell))
    ny = int(np.ceil((row["maxy"] - row["miny"]) / cell))
    if nx < 20 or ny < 20:
        return dict(base, ok=False, note="tile grid too small")
    cx = row["minx"] + (np.arange(nx) + 0.5) * cell
    cy = row["miny"] + (np.arange(ny) + 0.5) * cell
    GX, GY = np.meshgrid(cx, cy, indexing="ij")

    src = f"EPSG:{int(row['proj:epsg'])}"
    lon, lat = Transformer.from_crs(src, "EPSG:4326", always_xy=True).transform(
        float(GX.mean()), float(GY.mean()))
    cands = ref.candidates(lon, lat)
    if not cands:
        return dict(base, ok=False, note="no 3DEP tile covers this tile")

    tried = []
    for demkey, proj, year in cands:
        def to_dem(wkt, _gx=GX, _gy=GY):
            tr = Transformer.from_crs(src, wkt, always_xy=True)
            return tr.transform(_gx.ravel(), _gy.ravel())
        got = sample_dem_bilinear(demkey, src, to_dem)
        if isinstance(got, str):
            tried.append(f"{proj}:{got}")
            continue
        rz, rvalid, epsg = got
        R = rz.reshape(nx, ny)
        RV = rvalid.reshape(nx, ny)

        # Smooth the reference before differentiating. Differentiating a noisy surface and
        # then regressing the difference on that derivative correlates the regressor with
        # the noise in the response, which attenuates the fitted shift; smoothing limits it.
        Rf = np.where(RV, R, 0.0)
        wsum = uniform_filter(RV.astype(float), SMOOTH_CELLS, mode="nearest")
        Rs = np.where(wsum > 0,
                      uniform_filter(Rf, SMOOTH_CELLS, mode="nearest") / np.maximum(wsum, 1e-9),
                      np.nan)
        Hx, Hy = np.gradient(Rs, CELL_M, CELL_M)
        gmag = np.hypot(Hx, Hy)

        zz = z
        if null_test:
            # Zero-point test. Replace every point's elevation by the reference surface
            # sampled at that point's own horizontal position, keeping the real point
            # geometry -- density, gaps, tile edges -- intact. The true shift is then zero
            # by construction, so whatever comes back is the estimator's own bias.
            #
            # This is the one thing --synthetic-shift cannot check. That test translates the
            # cloud and measures the CHANGE in the answer, so any constant offset in the
            # estimator cancels out of it and passes unnoticed. A release-wide shift of the
            # same sign in every sortie is exactly that shape, so it has to be tested for
            # directly before being believed.
            def to_dem_pts(wkt, _x=x, _y=y):
                return Transformer.from_crs(src, wkt, always_xy=True).transform(_x, _y)
            pgot = sample_dem_bilinear(demkey, src, to_dem_pts)
            if isinstance(pgot, str):
                tried.append(f"{proj}:null-sample({pgot})")
                continue
            pz, pvalid, _ = pgot
            if int(pvalid.sum()) < 10000:
                tried.append(f"{proj}:null-few-points({int(pvalid.sum())})")
                continue
            x, y = x[pvalid], y[pvalid]
            zz = pz[pvalid] / M_PER_FTUS      # the pipeline expects ftUS
        G, usable = ground_grid(x, y, zz, row["minx"], row["miny"], np.nan_to_num(gmag))
        sel = usable & RV & np.isfinite(Hx) & np.isfinite(Hy) & np.isfinite(Rs)
        if int(sel.sum()) < MIN_FIT_CELLS:
            tried.append(f"{proj}:few-cells({int(sel.sum())})")
            continue

        dh = G[sel] * M_PER_FTUS - Rs[sel]
        hx, hy, gm = Hx[sel], Hy[sel], gmag[sel]
        cols = [np.ones(dh.size), hx, hy] + ([gm] if use_slope_term else [])
        A = np.column_stack(cols)
        fit = robust_fit(A, dh)
        if fit is None:
            tried.append(f"{proj}:fit-failed")
            continue
        coef, cov, nkeep, sig = fit
        se = np.sqrt(np.diag(cov))
        return dict(base, ok=True, dem_project=proj, dem_year=year, dem_epsg=epsg,
                    n=int(sel.sum()), n_fit=nkeep,
                    dz_m=round(float(coef[0]), 4),
                    dx_m=round(float(coef[1]), 4), dy_m=round(float(coef[2]), 4),
                    se_dx_m=round(float(se[1]), 4), se_dy_m=round(float(se[2]), 4),
                    beta_m=round(float(coef[3]), 4) if use_slope_term else None,
                    se_beta_m=round(float(se[3]), 4) if use_slope_term else None,
                    resid_sigma_m=round(sig, 4),
                    grad_p50=round(float(np.median(gm)), 4),
                    grad_p90=round(float(np.percentile(gm, 90)), 4),
                    # centred cross-products, the sufficient statistics for the pooled
                    # per-sortie fit with one vertical offset per tile
                    cross=cross_products(dh, cols[1:], A, coef),
                    cx=round(float(GX.mean()), 2), cy=round(float(GY.mean()), 2),
                    skipped=";".join(tried) or None)
    return dict(base, ok=False, note="no candidate reference usable", skipped=";".join(tried))


def cross_products(dh, shared, A, coef):
    """Within-tile centred cross-products of the shared regressors and the response.

    Centring within the tile is exactly what profiling out a per-tile intercept does, so
    summing these across a sortie's tiles and solving gives the pooled estimate with one
    vertical offset per tile, without keeping any cells.

    Outliers are trimmed first, using the tile's own fit, because that is where they are:
    unremoved vegetation, buildings the opening missed, and real change since the reference.
    """
    res = dh - A @ coef
    s = 1.4826 * np.median(np.abs(res - np.median(res)))
    keep = np.abs(res - np.median(res)) <= 3.0 * s if s > 0 else np.ones(dh.size, bool)
    cols = [c[keep] for c in shared]
    d = dh[keep]
    cols = [c - c.mean() for c in cols]
    d = d - d.mean()
    k = len(cols)
    return dict(n=int(keep.sum()),
                xx=[[round(float(cols[i] @ cols[j]), 6) for j in range(k)] for i in range(k)],
                xd=[round(float(cols[i] @ d), 6) for i in range(k)],
                dd=round(float(d @ d), 6))


def pooled(rows, use_slope_term):
    """Solve the shared (dx, dy[, beta]) for one sortie from the per-tile cross-products."""
    k = 3 if use_slope_term else 2
    XX = np.zeros((k, k)); XD = np.zeros(k); DD = 0.0; n = 0; ntile = 0
    for r in rows:
        c = r.get("cross")
        if not c or len(c["xd"]) != k:
            continue
        XX += np.array(c["xx"], float); XD += np.array(c["xd"], float)
        DD += c["dd"]; n += c["n"]; ntile += 1
    if ntile == 0 or n - ntile - k <= 0:
        return None
    try:
        XXi = np.linalg.inv(XX)
    except np.linalg.LinAlgError:
        return None
    coef = XXi @ XD
    rss = DD - XD @ coef
    dof = n - ntile - k          # one intercept per tile, plus the shared terms
    sigma2 = max(rss, 0.0) / dof
    cov = XXi * sigma2
    se = np.sqrt(np.diag(cov))
    # how well separated the two horizontal components are: 1 means orthogonal
    corr = XX[0, 1] / np.sqrt(XX[0, 0] * XX[1, 1]) if XX[0, 0] > 0 and XX[1, 1] > 0 else np.nan
    return dict(tiles=ntile, cells=n, dx=coef[0], dy=coef[1],
                se_dx=se[0], se_dy=se[1],
                beta=coef[2] if use_slope_term else np.nan,
                se_beta=se[2] if use_slope_term else np.nan,
                sigma=float(np.sqrt(sigma2)), gx_gy_corr=float(corr))


def relief_sample(index_path, n_per_sortie, seed):
    """Interior tiles carrying terrain, since flat tiles cannot inform a horizontal shift.

    Relief comes from the index elevation range, which in the noisier sorties is inflated by
    high noise rather than terrain, hence the upper bound as well as the lower one. The band
    is a screen, not a guarantee: what actually decides whether a tile informs the fit is the
    standard error it comes back with.

    Two properties of the draw matter for reuse and are easy to get wrong. It is seeded from
    a CRC of the sortie name rather than from `hash()`, which Python randomises per process,
    so the sample is reproducible across runs at all. And it takes a prefix of one fixed
    permutation rather than an independent choice of `n`, so raising `--n` EXTENDS the
    previous sample instead of redrawing it: every tile already measured stays in, and only
    the new ones need fetching.
    """
    d = pd.read_parquet(index_path, columns=["id", "sortie", "proj:bbox", "proj:epsg",
                                             "s3_path", "elevation:min", "elevation:max"])
    bb = np.array([json.loads(x) for x in d["proj:bbox"]], float)
    d["minx"], d["miny"], d["maxx"], d["maxy"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    d["relief_m"] = (d["elevation:max"] - d["elevation:min"]) * M_PER_FTUS
    xy = d["id"].str.extract(r"_X(\d+)_Y(\d+)")
    d["gx"], d["gy"] = xy[0].astype(int), xy[1].astype(int)
    keep = []
    for s, g in d.groupby("sortie", sort=False):
        present = set(zip(g["gx"], g["gy"]))
        inter = np.array([
            all((a + i, b + j) in present for i in (-1, 0, 1) for j in (-1, 0, 1)
                if (i, j) != (0, 0)) for a, b in zip(g["gx"], g["gy"])])
        gi = g[inter]
        band = gi[(gi["relief_m"] >= MIN_RELIEF_M) & (gi["relief_m"] <= MAX_RELIEF_M)]
        if len(band) < n_per_sortie:
            # Too few tiles in the band: fall back to the most relief available, taking the
            # highest first rather than sampling, since in a flat sortie every bit of
            # terrain counts. Capped so that noise-inflated outliers do not win the draw.
            band = gi.assign(_r=gi["relief_m"].clip(upper=MAX_RELIEF_M)) \
                     .sort_values("_r", ascending=False).head(n_per_sortie).drop(columns="_r")
            keep.append(band)
            continue
        rng = np.random.default_rng(zlib.crc32(s.encode()) + seed)
        order = rng.permutation(len(band))
        keep.append(band.iloc[order[:min(n_per_sortie, len(band))]])
    return pd.concat(keep, ignore_index=True)


def report(path, use_slope_term, synthetic, baseline=None, null_test=False):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows).drop_duplicates(subset="id", keep="last")
    good = d[d.ok == True] if "ok" in d else d
    pd.set_option("display.width", 220)
    print(f"\nusable tiles: {len(good)} of {len(d)} attempted")
    if "note" in d and d.ok.eq(False).any():
        print("\nrejections:")
        print(d[d.ok == False].groupby("note").size().to_string())

    recs = []
    for s, g in good.groupby("sortie"):
        p = pooled(g.to_dict("records"), use_slope_term)
        if p is None:
            continue
        # The formal standard error from the pooled fit treats each 2 m cell as an
        # independent observation, and they are nothing of the kind: the residuals are
        # strongly correlated over tens of metres, so the effective sample size is far
        # below the cell count and the formal error is optimistic by an order of magnitude.
        # Inflate it by the scatter of the per-tile estimates about the pooled value,
        # which is the standard chi-square-per-degree-of-freedom correction and also
        # measures how badly the independence assumption fails.
        infl = {}
        for ax in ("dx", "dy"):
            est = g[f"{ax}_m"].to_numpy(float)
            sef = g[f"se_{ax}_m"].to_numpy(float)
            m = np.isfinite(est) & np.isfinite(sef) & (sef > 0)
            if m.sum() >= 2:
                chi2 = float((((est[m] - p[ax]) / sef[m]) ** 2).sum()) / (m.sum() - 1)
                infl[ax] = max(np.sqrt(chi2), 1.0)
            else:
                infl[ax] = np.nan
        se_dx = p["se_dx"] * infl["dx"]
        se_dy = p["se_dy"] * infl["dy"]
        # model-free alternative: robust scatter of the per-tile estimates over sqrt(n)
        sd_x = float(1.4826 * np.median(np.abs(g["dx_m"] - np.median(g["dx_m"]))))
        sd_y = float(1.4826 * np.median(np.abs(g["dy_m"] - np.median(g["dy_m"]))))
        se_tile = max(sd_x, sd_y) / np.sqrt(max(p["tiles"], 1))
        mag = float(np.hypot(p["dx"], p["dy"]))
        se_mag = float(np.hypot(p["dx"] * se_dx, p["dy"] * se_dy) / max(mag, 1e-9))
        se_use = max(se_mag, se_tile)
        recs.append(dict(sortie=s, tiles=p["tiles"], cells=p["cells"],
                         dx_m=p["dx"], dy_m=p["dy"], se_dx=se_dx, se_dy=se_dy,
                         se_formal_dx=p["se_dx"], infl_dx=infl["dx"], infl_dy=infl["dy"],
                         sd_tile_x=sd_x, sd_tile_y=sd_y, se_tile=se_tile,
                         mag_m=mag, se_mag=se_use, snr=mag / max(se_use, 1e-9),
                         beta_m=p["beta"], se_beta=p["se_beta"],
                         sigma_m=p["sigma"], gx_gy_corr=p["gx_gy_corr"],
                         grad_p50=float(g["grad_p50"].median()),
                         ref_years="/".join(str(int(y)) for y in sorted(set(g["dem_year"])))))
    r = pd.DataFrame(recs).sort_values("sortie").reset_index(drop=True)
    if not len(r):
        print("no sortie had enough usable tiles")
        return r

    print("\n" + "=" * 120)
    print("HORIZONTAL SHIFT OF INSIGHTS RELATIVE TO THE 3DEP 1 m DEM")
    print("pooled per sortie, one vertical offset per tile, shared (dx, dy)")
    print("=" * 120)
    cols = ["sortie", "tiles", "cells", "dx_m", "se_dx", "dy_m", "se_dy", "mag_m",
            "se_mag", "snr", "infl_dx", "sd_tile_x", "grad_p50", "sigma_m", "gx_gy_corr",
            "ref_years"]
    print(r[cols].to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.3f}"))
    print("\n  dx, dy      translation to APPLY to INSIGHTS to align it with the reference, m")
    print("  se_*        standard error, the formal one inflated by the between-tile scatter")
    print("  infl_dx     that inflation factor. It is large because cells inside a tile are")
    print("              spatially correlated, so the formal error is far too small; treat a")
    print("              raw per-cell standard error from this fit as meaningless")
    print("  sd_tile_x   robust scatter of the per-tile estimates, m: the honest spread")
    print("  mag, snr    magnitude of the shift and its ratio to its own standard error;")
    print("              snr below about 3 means the terrain did not constrain the estimate")
    print("  grad_p50    median terrain gradient of the cells used: the leverage available")
    print("  sigma_m     residual scatter of the fit, m")
    print("  gx_gy_corr  correlation between the two gradient components; near +/-1 means")
    print("              the two horizontal components cannot be separated in this terrain")

    if synthetic:
        tx, ty = synthetic
        print(f"\nSYNTHETIC TEST: the point clouds were translated by "
              f"({tx:+.2f}, {ty:+.2f}) m before fitting.")
        print("  Translating INSIGHTS by T leaves a correction of (base - T), so recovery is")
        print("  measured as base minus this run, against T. It is NOT this run against T:")
        print("  the tiles carry a real shift already, and comparing to zero would fold it in.")
        if baseline is None:
            print("  No --baseline given, so only the raw estimates are shown. Run without")
            print("  --synthetic-shift first, then pass that run's _by_sortie.csv.")
            print(r[["sortie", "dx_m", "se_dx", "dy_m", "se_dy", "snr"]].to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
            return r
        b = pd.read_csv(baseline).set_index("sortie")
        j = r.set_index("sortie").join(b[["dx_m", "dy_m", "snr"]], rsuffix="_base",
                                       how="inner")
        j["rec_x"] = j["dx_m_base"] - j["dx_m"]
        j["rec_y"] = j["dy_m_base"] - j["dy_m"]
        j["err_x"] = j["rec_x"] - tx
        j["err_y"] = j["rec_y"] - ty
        ok = j[(j.snr >= 3) & (j.snr_base >= 3)]
        print(j[["dx_m", "dy_m", "rec_x", "rec_y", "err_x", "err_y", "snr"]].to_string(
            float_format=lambda v: f"{v:.3f}"))
        if len(ok):
            print(f"\n  over the {len(ok)} well-conditioned sorties:")
            print(f"    recovered ({ok.rec_x.median():+.3f}, {ok.rec_y.median():+.3f}) m "
                  f"against an applied ({tx:+.3f}, {ty:+.3f}) m")
            print(f"    error     ({ok.err_x.median():+.3f}, {ok.err_y.median():+.3f}) m "
                  f"median, worst ({ok.err_x.abs().max():.3f}, {ok.err_y.abs().max():.3f}) m")
            sc = [ok.rec_x.median() / tx if tx else np.nan,
                  ok.rec_y.median() / ty if ty else np.nan]
            print(f"    scale     {sc[0]:.3f}, {sc[1]:.3f} of the applied shift; below 1 is")
            print(f"              attenuation from the noisy-gradient regressor, and the")
            print(f"              measured shifts should be divided by it to be unbiased")
        return r

    if null_test:
        print("\nZERO-POINT TEST: point elevations were replaced by the reference surface,")
        print("  so the true shift is zero and everything below is estimator bias.")
        # Summarise over EVERY sortie, not the well-conditioned ones. Filtering a null
        # test by significance selects for the sorties whose bias is most distinguishable
        # from zero, which is the opposite of what needs reporting: the question is how
        # large the bias is anywhere, not where it is statistically detectable.
        print(f"  over all {len(r)} sorties: |dx| at most {r.dx_m.abs().max():.3f} m "
              f"(median {r.dx_m.abs().median():.3f}), |dy| at most "
              f"{r.dy_m.abs().max():.3f} m (median {r.dy_m.abs().median():.3f}),")
        print(f"  worst magnitude {r.mag_m.max():.3f} m.")
        print("  Compare against the measured shifts. A bias much smaller than those means")
        print("  they are real; a bias of the same size means they are not.")
        print("  Note that the flat sorties come out near zero here while failing on real")
        print("  data: against a noiseless surface the fit is well determined even at low")
        print("  gradient, so their real-data failure is noise, not the estimator.")
        return r

    strong = r[r.snr >= 3]
    print(f"\n{len(strong)} of {len(r)} sorties have terrain enough to resolve a shift "
          f"(snr >= 3): {', '.join(strong['sortie']) if len(strong) else 'none'}")
    if len(strong):
        print(f"  |shift| {strong.mag_m.min():.2f} to {strong.mag_m.max():.2f} m, "
              f"median {strong.mag_m.median():.2f} m")
        print(f"  dx {strong.dx_m.min():+.2f} to {strong.dx_m.max():+.2f} m, "
              f"dy {strong.dy_m.min():+.2f} to {strong.dy_m.max():+.2f} m")
        agree = (np.sign(strong.dx_m).nunique() == 1, np.sign(strong.dy_m).nunique() == 1)
        print(f"  sign consistent across sorties: dx {agree[0]}, dy {agree[1]} -- a common "
              f"sign would point to a release-wide\n  offset, mixed signs to a per-sortie one")
    weak = r[r.snr < 3]
    if len(weak):
        print(f"\n{len(weak)} sorties lack the terrain to resolve a shift: "
              f"{', '.join(weak['sortie'])}")
        print("  Their estimates are not evidence of good registration, only of low leverage;")
        print(f"  median terrain gradient there is {weak.grad_p50.median():.3f} "
              f"against {strong.grad_p50.median():.3f} where it worked."
              if len(strong) else "")
    if use_slope_term and r["beta_m"].notna().any():
        pred = -0.4 * CELL_M
        print(f"\nGround-estimator bias term: fitted beta median "
              f"{r['beta_m'].median():.3f} m per unit gradient, against {pred:.2f} m "
              f"predicted for\n  a 10th-percentile proxy on a {CELL_M:.0f} m cell. Agreement "
              f"means the term is absorbing the\n  effect it was added for rather than "
              f"soaking up something else.")
    return r


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(repo, "data", "lidar", "v1", "stac",
                                                    "index", "items.parquet"))
    ap.add_argument("--n", type=int, default=15, help="tiles per sortie")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sortie", action="append")
    ap.add_argument("--synthetic-shift", default=None,
                    help="DX,DY in metres, applied to the points before fitting, to "
                         "validate the estimator against known truth")
    ap.add_argument("--no-slope-term", action="store_true",
                    help="drop the |grad| regressor that absorbs the ground-proxy bias")
    ap.add_argument("--null-test", action="store_true",
                    help="replace point elevations with the reference surface, so the true "
                         "shift is zero and anything recovered is estimator bias")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--baseline", default=None,
                    help="a previous run's _by_sortie.csv, to measure "
                         "synthetic-shift recovery against")
    ap.add_argument("--cache-dir", default=None,
                    help="cache downloaded tiles here, so repeated runs (e.g. the "
                         "synthetic-shift validation) do not re-fetch them")
    ap.add_argument("--out", default=os.path.join(here, "out", "coregister_3dep.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    syn = tuple(float(v) for v in a.synthetic_shift.split(",")) if a.synthetic_shift else None
    use_slope = not a.no_slope_term

    if not a.report_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print("indexing 3DEP 1 m projects", file=sys.stderr)
        ref = C3.Reference(C3.build_index(
            os.path.join(os.path.dirname(a.out), "dem_index.json")))
        samp = relief_sample(a.index, a.n, a.seed)
        if a.sortie:
            samp = samp[samp["sortie"].isin(a.sortie)]
        print(f"  median relief of sampled tiles: {samp['relief_m'].median():.1f} m",
              file=sys.stderr)
        done = set()
        if os.path.exists(a.out):
            done = {json.loads(l).get("id") for l in open(a.out)}
            print(f"resuming, {len(done)} tiles already done", file=sys.stderr)
        todo = samp[~samp["id"].isin(done)]
        print(f"co-registering {len(todo)} tiles with {a.workers} workers", file=sys.stderr)
        errs = []
        with open(a.out, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
            if a.cache_dir:
                os.makedirs(a.cache_dir, exist_ok=True)
            futs = {ex.submit(process, r, ref, syn, use_slope, a.cache_dir,
                              a.null_test): r["id"]
                    for _, r in todo.iterrows()}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    fh.write(json.dumps(f.result()) + "\n"); fh.flush()
                except Exception as e:
                    errs.append((futs[f], repr(e)[:200]))
                if i % 25 == 0:
                    print(f"  {i}/{len(futs)}  ({len(errs)} errors)", file=sys.stderr,
                          flush=True)
        for tid, msg in errs:
            print(f"  err {tid} {msg}")

    r = report(a.out, use_slope, syn, a.baseline, a.null_test)
    if len(r):
        r.to_csv(a.out.replace(".jsonl", "_by_sortie.csv"), index=False)
        print(f"\nwrote {a.out} and {a.out.replace('.jsonl', '_by_sortie.csv')}")


if __name__ == "__main__":
    main()
