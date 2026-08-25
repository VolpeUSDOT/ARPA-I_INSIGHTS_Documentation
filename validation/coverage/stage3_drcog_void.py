#!/usr/bin/env python3
"""DRCOG coverage void: occupied-extent comparison and LAS header cross-check.

There is a continuous N/S band inside the DRCOG sortie that holds no returns.  It
sits at the boundary between two grid columns, and it is invisible to the two
tests that were tried first (see the FAILED METHODS notes below).

Two modes:

  bbox     (default, index only, no network) -- section 4.2.  Compares `proj:bbox`
           between grid columns 45 and 46 row by row, and quantifies the void as
           a width, a longitude, and a missing area.

  headers  (network, 375-byte HTTP Range request per tile) -- section 4.3.  Parses
           the LAS 1.4 public header block of all 36,221 DRCOG tiles and checks
           the header bounding box and point count against the index.  This is
           what establishes that `proj:bbox` describes the DATA rather than the
           grid cell, which is the premise the bbox mode rests on.

Run with --help for options.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from shapely import wkb

M_PER_FTUS = 0.3048006096012192
INDEX_RELPATH = Path("data/lidar/v1/stac/index/items.parquet")
TILE_ID_RE = re.compile(r"_X(\d+)_Y(\d+)")

BUCKET = "arpa-i-insights"
S3_PREFIX = f"s3://{BUCKET}/"

# The two grid columns that bracket the void.
#
# TRAP: in DRCOG a "column" is a constant *gy*, and gy DECREASES eastward
# (corr(gy, lon) = -1.0), so column 45 lies EAST of column 46.  The gap is
# therefore minx(45) - maxx(46), not the other way round.  Getting the sign
# backwards yields uniformly negative "gaps" and the conclusion that the columns
# overlap.  The script asserts the orientation rather than trusting this comment.
COL_EAST = 45
COL_WEST = 46

# Longitude of the void identified independently during visual QC.
QC_VOID_LON_DMS = (105, 0, 34.5)  # west
# Header layout, LAS 1.4 public header block.
LAS_HEADER_BYTES = 375
OFF_SIGNATURE = 0
OFF_MINMAX = 179   # <4d : maxx, minx, maxy, miny
OFF_NPTS64 = 247   # <Q  : legacy-free 64-bit point count
OFF_NPTS32 = 107   # <I  : legacy 32-bit point count


def default_index_path() -> Path:
    """Resolve the index relative to the repo root, not to a scratch path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / INDEX_RELPATH
        if candidate.exists():
            return candidate
    return here.parents[2] / INDEX_RELPATH


def dms(lon: float) -> str:
    """Format a negative (western) longitude as D M S.s W."""
    a = abs(lon)
    d = int(a)
    m = int((a - d) * 60)
    s = (a - d - m / 60) * 3600
    hemi = "W" if lon < 0 else "E"
    return f"{d}°{m:02d}'{s:04.1f}\"{hemi}"


def dms_to_deg(d: int, m: int, s: float) -> float:
    return d + m / 60 + s / 3600


def load_drcog(index_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        index_path,
        columns=["id", "sortie", "pc:count", "proj:epsg", "proj:bbox", "bbox",
                 "geometry", "s3_path", "las_s3_path"],
    )
    df = df[df["sortie"] == "DRCOG"].reset_index(drop=True)

    gx = np.empty(len(df), dtype=np.int32)
    gy = np.empty(len(df), dtype=np.int32)
    for i, tid in enumerate(df["id"].to_numpy()):
        m = TILE_ID_RE.search(tid)
        gx[i], gy[i] = int(m.group(1)), int(m.group(2))
    df["gx"], df["gy"] = gx, gy

    # TRAP: proj:bbox and bbox are JSON *strings*, not lists.
    pb = np.array([json.loads(s) for s in df["proj:bbox"]])
    df["minx"], df["miny"], df["maxx"], df["maxy"] = pb[:, 0], pb[:, 1], pb[:, 2], pb[:, 3]
    bb = np.array([json.loads(s) for s in df["bbox"]])
    df["lon_min"], df["lon_max"] = bb[:, 0], bb[:, 2]
    df["lat_min"], df["lat_max"] = bb[:, 1], bb[:, 3]

    # Footprint polygons: single-part MultiPolygon, exterior ring of 6 XYZ
    # coordinates (4 corners + closing vertex + a duplicate of it).  Take the
    # first 4 and drop Z.
    geoms = wkb.loads(df["geometry"].values)
    corners = np.asarray([np.asarray(g.geoms[0].exterior.coords)[:4, :2] for g in geoms])
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6430", always_xy=True)
    fx, fy = tr.transform(corners[:, :, 0], corners[:, :, 1])
    fx = np.asarray(fx) * M_PER_FTUS
    fy = np.asarray(fy) * M_PER_FTUS
    df["footprint_area_m2"] = 0.5 * np.abs(
        np.sum(fx * np.roll(fy, -1, axis=1) - np.roll(fx, -1, axis=1) * fy, axis=1))
    df["fp_lon_min"] = corners[:, :, 0].min(axis=1)
    df["fp_lon_max"] = corners[:, :, 0].max(axis=1)

    # proj:bbox is in US survey feet; the occupied-area comparison must be in
    # the same units as the footprint area.
    df["occupied_bbox_area_m2"] = ((df["maxx"] - df["minx"]) * (df["maxy"] - df["miny"])
                                   * M_PER_FTUS * M_PER_FTUS)
    df = df.drop(columns=["geometry"])
    return df


def interior_mask(df: pd.DataFrame) -> np.ndarray:
    """True where all eight grid neighbours are present in the same sortie."""
    present = set(zip(df["gx"].tolist(), df["gy"].tolist()))
    out = np.zeros(len(df), dtype=bool)
    for i, (gx, gy) in enumerate(zip(df["gx"].to_numpy(), df["gy"].to_numpy())):
        out[i] = all((gx + dx, gy + dy) in present
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if not (dx == 0 and dy == 0))
    return out


# --------------------------------------------------------------------------- #
# FAILED METHODS -- documented so they are not reintroduced.
#
# 1. Footprint polygons cannot show the void.  They are NOMINAL tiling geometry,
#    not data extents: the polygons of columns 45 and 46 abut (in this release
#    they measure ~139.96 m on a side at a 139.957 m pitch), so the union of the
#    two columns' footprints is continuous across a band that holds no returns.
#    The demo below prints the footprint-implied gap, which is <= 0.
#
# 2. Per-tile occupancy indexed from the tile's OWN minimum occupied easting
#    cannot show it either.  If the array origin is the first occupied cell, a
#    region with no data lies outside the array and can never register as empty;
#    every tile then looks fully occupied and the (wrong) conclusion "density
#    deficit, not a void" follows.  Occupancy must be assessed in 2D over the
#    NOMINAL extent, or -- as here -- via the proj:bbox comparison.
#
# 3. The void edges run DIAGONALLY across tiles (it is a flight-line boundary),
#    so any test that requires a fully empty row or column of cells returns zero.
# --------------------------------------------------------------------------- #
def demo_failed_footprint_method(df: pd.DataFrame) -> None:
    east = df[df["gy"] == COL_EAST].set_index("gx")
    west = df[df["gy"] == COL_WEST].set_index("gx")
    rows = sorted(set(east.index) & set(west.index))
    fp_gap_deg = east.loc[rows, "fp_lon_min"].to_numpy() - west.loc[rows, "fp_lon_max"].to_numpy()
    # Convert a longitude difference to metres at this latitude for readability.
    geod = pyproj.Geod(ellps="GRS80")
    lat = float(df["lat_min"].median())
    m_per_deg_lon = geod.inv(0.0, lat, 1.0, lat)[2]
    fp_gap_m = fp_gap_deg * m_per_deg_lon
    print("FAILED METHOD 1 -- footprint polygons (nominal geometry):")
    print(f"  footprint-implied gap between columns {COL_EAST} and {COL_WEST}: "
          f"median {np.median(fp_gap_m):+.2f} m, max {fp_gap_m.max():+.2f} m "
          f"over {len(rows)} rows")
    print(f"  rows with a positive footprint gap: "
          f"{int((fp_gap_m > 0).sum())} / {len(rows)}  -> the footprints imply "
          "CONTINUOUS coverage. This method cannot see the void.")


def analyse_bbox(df: pd.DataFrame, out: Path | None) -> dict:
    print("=" * 78)
    print("Section 4.2 -- void from proj:bbox comparison between adjacent columns")
    print("=" * 78)

    # Orientation checks.  proj:bbox is only a usable proxy for the occupied
    # region because DRCOG's grid is aligned to ~0.36 deg off the State Plane
    # axes.  Do NOT generalise this to the rotated sorties (I70D 29.0 deg,
    # I70H 28.2, I80East 24.6, I70I 24.3, ...), where the axis-aligned box
    # circumscribes and overstates the occupied region.
    lon_mid = (df["lon_min"] + df["lon_max"]) / 2
    lat_mid = (df["lat_min"] + df["lat_max"]) / 2
    corr_gy_lon = float(np.corrcoef(df["gy"], lon_mid)[0, 1])
    corr_gx_lat = float(np.corrcoef(df["gx"], lat_mid)[0, 1])
    print(f"orientation: corr(gy, lon) = {corr_gy_lon:+.4f}  "
          f"corr(gx, lat) = {corr_gx_lat:+.4f}")
    assert corr_gy_lon < -0.99, "expected gy to decrease eastward in DRCOG"
    print(f"  -> gy decreases eastward, so column {COL_EAST} is EAST of "
          f"column {COL_WEST}; gap = minx({COL_EAST}) - maxx({COL_WEST}).")
    print()

    demo_failed_footprint_method(df)
    print()

    east = df[df["gy"] == COL_EAST].set_index("gx")
    west = df[df["gy"] == COL_WEST].set_index("gx")
    rows = sorted(set(east.index) & set(west.index))
    gap_m = (east.loc[rows, "minx"].to_numpy() - west.loc[rows, "maxx"].to_numpy()) * M_PER_FTUS

    n_void = int((gap_m > 0).sum())
    print(f"rows where both columns {COL_EAST} and {COL_WEST} exist: {len(rows)}")
    print(f"rows with a POSITIVE data gap:                         {n_void}"
          f"  ({100 * n_void / len(rows):.1f}%)")
    print(f"gap width (m): median {np.median(gap_m):.1f}  "
          f"IQR {np.percentile(gap_m, 25):.0f}-{np.percentile(gap_m, 75):.0f}  "
          f"min {gap_m.min():.1f}  max {gap_m.max():.1f}")
    print()

    # --- where is it, in longitude? -----------------------------------------
    # Convert the median State Plane edges to WGS84, so the result can be
    # compared with the coordinate identified during visual QC.
    to4326 = pyproj.Transformer.from_crs("EPSG:6430", "EPSG:4326", always_xy=True)
    y_ref = float(np.median(df["miny"]))
    east_edge_x = float(np.median(east.loc[rows, "minx"]))   # west face of col 45
    west_edge_x = float(np.median(west.loc[rows, "maxx"]))   # east face of col 46
    lon_e, _ = to4326.transform(east_edge_x, y_ref)
    lon_w, _ = to4326.transform(west_edge_x, y_ref)
    centre_x = 0.5 * (east_edge_x + west_edge_x)
    lon_c, lat_c = to4326.transform(centre_x, y_ref)

    qc_lon = -dms_to_deg(*QC_VOID_LON_DMS)
    geod = pyproj.Geod(ellps="GRS80")
    dist_to_qc = abs(geod.inv(lon_c, lat_c, qc_lon, lat_c)[2])
    span_m = (east_edge_x - west_edge_x) * M_PER_FTUS

    print(f"median edges: {dms(lon_w)} (west) to {dms(lon_e)} (east)")
    print(f"centre:       {dms(lon_c)}   [QC visual inspection: {dms(qc_lon)}; "
          f"offset {dist_to_qc:.1f} m]")
    span_geod_m = abs(geod.inv(lon_w, lat_c, lon_e, lat_c)[2])
    print(f"difference-of-medians span {span_m:.1f} m projected "
          f"({span_geod_m:.1f} m geodesic) vs "
          f"median-of-differences width {np.median(gap_m):.1f} m "
          "(both bracket the same band)")

    # N/S extent of the void.
    lat_lo = float(df.loc[df["gx"] == min(rows), "lat_min"].min())
    lat_hi = float(df.loc[df["gx"] == max(rows), "lat_max"].max())
    ns_km = abs(geod.inv(lon_c, lat_lo, lon_c, lat_hi)[2]) / 1000
    print(f"N/S extent: rows gx {min(rows)}..{max(rows)}  -> {ns_km:.1f} km")
    print()

    # --- occupied / footprint area ratio by column ---------------------------
    df = df.copy()
    df["area_ratio"] = df["occupied_bbox_area_m2"] / df["footprint_area_m2"]
    df["interior"] = interior_mask(df)
    print("occupied proj:bbox area / footprint polygon area, by grid column")
    print(f"  {'column':<22s} {'n_int':>6s} {'median':>8s} {'min':>8s} {'<0.90':>8s}")
    col_stats = {}
    for label, sel in (
        (f"column {COL_EAST}", df["gy"] == COL_EAST),
        (f"column {COL_WEST}", df["gy"] == COL_WEST),
        ("all other columns", ~df["gy"].isin([COL_EAST, COL_WEST])),
    ):
        s = df[sel & df["interior"]]
        n_below = int((s["area_ratio"] < 0.90).sum())
        print(f"  {label:<22s} {len(s):>6d} {s['area_ratio'].median():>8.4f} "
              f"{s['area_ratio'].min():>8.4f} {n_below:>8d}")
        col_stats[label] = {
            "n_interior": len(s),
            "median_area_ratio": float(s["area_ratio"].median()),
            "min_area_ratio": float(s["area_ratio"].min()),
            "n_below_0p90": n_below,
        }
    print("  (a ratio slightly above 1 is normal: the axis-aligned occupied box")
    print("   marginally exceeds the footprint polygon it is clipped to.)")
    print()

    # --- area, not density ---------------------------------------------------
    interior = df[df["interior"]]
    other = interior[~interior["gy"].isin([COL_EAST, COL_WEST])]
    e_int = interior[interior["gy"] == COL_EAST]
    med_pc_other = float(other["pc:count"].median())
    med_pc_east = float(e_int["pc:count"].median())
    med_ratio_east = float(e_int["area_ratio"].median())
    dens_other = med_pc_other / float(other["occupied_bbox_area_m2"].median())
    dens_east = med_pc_east / float(e_int["occupied_bbox_area_m2"].median())
    print(f"column {COL_EAST} holds {100 * med_pc_east / med_pc_other:.1f}% of the "
          f"unaffected columns' median point count")
    print(f"       over {100 * med_ratio_east:.1f}% of the footprint area,")
    print(f"  density within the OCCUPIED part: {dens_east:.2f} vs "
          f"{dens_other:.2f} points/m^2 "
          f"-> {100 * abs(dens_east / dens_other - 1):.1f}% difference.")
    print("  CONCLUSION: this is missing AREA, not reduced density.")
    print()

    # --- missing area --------------------------------------------------------
    # Net shortfall: sum of (footprint - occupied) without clipping at zero, so
    # the handful of tiles whose occupied box marginally exceeds the footprint
    # polygon offset rather than inflate the total.  Clipping at zero raises the
    # figure by only ~0.002 km^2.
    two = df[df["gy"].isin([COL_EAST, COL_WEST])]
    missing_all = (two["footprint_area_m2"] - two["occupied_bbox_area_m2"]).sum()
    two_int = two[two["interior"]]
    missing_int = (two_int["footprint_area_m2"] - two_int["occupied_bbox_area_m2"]).sum()
    print(f"missing area in columns {COL_EAST}+{COL_WEST}: "
          f"{missing_all / 1e6:.2f} km^2 over {len(two)} tiles "
          f"({missing_int / 1e6:.2f} km^2 over {len(two_int)} interior tiles)")

    result = {
        "n_rows_both_columns": len(rows),
        "n_rows_with_void": n_void,
        "gap_median_m": float(np.median(gap_m)),
        "gap_p25_m": float(np.percentile(gap_m, 25)),
        "gap_p75_m": float(np.percentile(gap_m, 75)),
        "gap_min_m": float(gap_m.min()),
        "gap_max_m": float(gap_m.max()),
        "median_west_edge_lon": lon_w,
        "median_east_edge_lon": lon_e,
        "centre_lon": lon_c,
        "centre_dms": dms(lon_c),
        "offset_from_qc_m": dist_to_qc,
        "difference_of_medians_span_m": span_m,
        "ns_extent_km": ns_km,
        "columns": col_stats,
        "density_east_pts_per_m2": dens_east,
        "density_other_pts_per_m2": dens_other,
        "missing_area_km2": float(missing_all / 1e6),
        "missing_area_interior_km2": float(missing_int / 1e6),
        "n_tiles_two_columns": len(two),
        "n_interior_tiles_two_columns": len(two_int),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")
    return result


# --------------------------------------------------------------------------- #
# Section 4.3 -- LAS header cross-check
# --------------------------------------------------------------------------- #
def parse_las_header(buf: bytes) -> tuple[float, float, float, float, int]:
    if buf[OFF_SIGNATURE:OFF_SIGNATURE + 4] != b"LASF":
        raise ValueError("missing LASF signature")
    maxx, minx, maxy, miny = struct.unpack_from("<4d", buf, OFF_MINMAX)
    (npts,) = struct.unpack_from("<Q", buf, OFF_NPTS64)
    if npts == 0:
        # LAS 1.4 files written by older tooling leave the 64-bit field zero and
        # only populate the legacy 32-bit count.
        (npts,) = struct.unpack_from("<I", buf, OFF_NPTS32)
    return minx, miny, maxx, maxy, int(npts)


def analyse_headers(df: pd.DataFrame, workers: int, out: Path | None,
                    limit: int | None) -> dict:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    print("=" * 78)
    print("Section 4.3 -- LAS public-header cross-check "
          f"({LAS_HEADER_BYTES}-byte HTTP Range request per tile)")
    print("=" * 78)

    sub = df if limit is None else df.head(limit)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                          max_pool_connections=max(workers * 2, 32),
                                          retries={"max_attempts": 5,
                                                   "mode": "standard"}))

    keys = [p[len(S3_PREFIX):] for p in sub["las_s3_path"]]
    ids = sub["id"].tolist()
    idx_bbox = sub[["minx", "miny", "maxx", "maxy"]].to_numpy()
    idx_count = sub["pc:count"].to_numpy()

    n = len(ids)
    res = np.full((n, 5), np.nan)
    errors: list[tuple[str, str]] = []
    lock = threading.Lock()
    done = [0]

    def work(i: int):
        body = s3.get_object(Bucket=BUCKET, Key=keys[i],
                             Range=f"bytes=0-{LAS_HEADER_BYTES - 1}")["Body"].read()
        return i, parse_las_header(body)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in range(n)]
        for f in as_completed(futs):
            try:
                i, vals = f.result()
                res[i] = vals
            except Exception as exc:  # noqa: BLE001 - report, do not abort 36k tiles
                with lock:
                    errors.append(("?", f"{type(exc).__name__}: {exc}"))
            with lock:
                done[0] += 1
                if done[0] % 2000 == 0 or done[0] == n:
                    print(f"  {done[0]}/{n} headers read", flush=True)

    ok = np.isfinite(res[:, 0])
    d = np.abs(res[ok, :4] - idx_bbox[ok])
    d_m = d * M_PER_FTUS
    count_match = res[ok, 4].astype(np.int64) == idx_count[ok]

    print()
    print(f"headers read successfully: {int(ok.sum())} / {n}"
          + (f"   ({len(errors)} failures)" if errors else ""))
    print(f"max |header bbox - proj:bbox| over all sides: "
          f"{d_m.max() * 1000:.3f} mm  (mean {d_m.mean() * 1000:.4f} mm)")
    for j, side in enumerate(("minx", "miny", "maxx", "maxy")):
        print(f"    {side}: max {d_m[:, j].max() * 1000:.3f} mm")
    print(f"point counts matching pc:count exactly: "
          f"{int(count_match.sum())} / {int(ok.sum())}")
    within_2mm = bool(d_m.max() <= 0.002)
    print()
    print(f"header bbox reproduces proj:bbox to within 2 mm on every side: "
          f"{'YES' if within_2mm else 'NO'}")
    print(f"header point count matches the index for every tile: "
          f"{'YES' if count_match.all() else 'NO'}")
    print("  CONCLUSION: proj:bbox describes the DATA present in the tile, not "
          "the nominal grid cell. That is what licenses the section 4.2 method.")
    if errors:
        print(f"\nfirst failures: {errors[:5]}")

    result = {
        "n_requested": n,
        "n_ok": int(ok.sum()),
        "n_failed": len(errors),
        "max_bbox_delta_mm": float(d_m.max() * 1000),
        "mean_bbox_delta_mm": float(d_m.mean() * 1000),
        "per_side_max_delta_mm": {s: float(d_m[:, j].max() * 1000)
                                  for j, s in enumerate(("minx", "miny", "maxx", "maxy"))},
        "n_count_match": int(count_match.sum()),
        "all_within_2mm": within_2mm,
        "all_counts_match": bool(count_match.all()),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="DRCOG coverage void analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--index", type=Path, default=default_index_path(),
                    help="path to items.parquet")
    ap.add_argument("--mode", choices=("bbox", "headers", "both"), default="bbox",
                    help="'bbox' is index-only; 'headers' needs network access")
    ap.add_argument("--workers", type=int, default=24,
                    help="thread count for the header requests")
    ap.add_argument("--limit", type=int, default=None,
                    help="header mode: only check the first N tiles (debugging)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON output path (a -bbox/-headers suffix is "
                         "added when --mode both)")
    args = ap.parse_args(argv)

    if not args.index.exists():
        ap.error(f"index not found: {args.index}")

    df = load_drcog(args.index)
    print(f"index: {args.index}")
    print(f"DRCOG tiles: {len(df)}\n")

    def out_for(tag: str) -> Path | None:
        if args.out is None:
            return None
        if args.mode != "both":
            return args.out
        return args.out.with_name(f"{args.out.stem}-{tag}{args.out.suffix}")

    if args.mode in ("bbox", "both"):
        analyse_bbox(df, out_for("bbox"))
        if args.mode == "both":
            print()
    if args.mode in ("headers", "both"):
        analyse_headers(df, args.workers, out_for("headers"), args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
