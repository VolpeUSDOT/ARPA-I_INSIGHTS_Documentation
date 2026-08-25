"""Test the manuscript's unmeasured claim that relative geometry is 'considerably
better' than the 1.71 m absolute horizontal RMSE.

The claim is currently argued rather than measured. Because 9.1% of the release area is
covered by more than one sortie, the overlap regions provide independent looks at the
same ground, which is exactly the relative regime the claim is about.

Method
------
Find tiles from DIFFERENT sorties whose occupied extents (proj:bbox) overlap, then over
the overlap compare the two independently-acquired surfaces cell by cell:

  * 2 m grid over the overlap rectangle.
  * Per cell and per sortie, a ground proxy = 10th percentile of Z.
  * Keep a cell only if BOTH sorties have >= 30 points in it AND the within-cell
    spread (p90-p10) is under 0.5 m in BOTH. That restricts the comparison to flat,
    hard, fully-sampled ground and excludes vegetation, building edges, and any cell
    where the surface itself is ambiguous.
  * The per-cell elevation difference then measures relative vertical agreement
    between two independent looks.

What this does and does not measure: it is a BETWEEN-SORTIE relative check over
co-observed ground. It does not measure within-sortie relative geometry, and it says
nothing directly about horizontal agreement, which cannot be recovered from flat ground
without matched features. Both limitations are reported with the result.

Same-metro sorties share a projected CRS (Denver 6430, Salt Lake 6626), so only
same-CRS pairs are compared and no reprojection is involved.
"""

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import glob
import laspy
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

M_PER_FTUS = 0.3048006096012192
CELL_M = 2.0
MIN_PTS_PER_CELL = 30
MAX_SPREAD_M = 0.5

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=16))


def load_index(path):
    d = pd.read_parquet(path, columns=["id", "sortie", "proj:bbox", "proj:epsg",
                                       "pc:count", "s3_path"])
    bb = np.array([json.loads(x) for x in d["proj:bbox"]], dtype=float)
    d["minx"], d["miny"], d["maxx"], d["maxy"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    return d


def find_pairs(d, min_overlap_m2, max_pairs, seed):
    """Cross-sortie tile pairs whose occupied extents overlap enough to compare."""
    pairs = []
    for epsg, grp in d.groupby("proj:epsg"):
        # bucket by coarse grid cell so we only test plausibly-near tiles
        step = 500 / M_PER_FTUS
        grp = grp.assign(bx=(grp["minx"] / step).astype(int),
                         by=(grp["miny"] / step).astype(int))
        buckets = {}
        for r in grp.itertuples():
            for dx in (-1, 0):
                for dy in (-1, 0):
                    buckets.setdefault((r.bx + dx, r.by + dy), []).append(r)
        seen = set()
        for key, rows in buckets.items():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    if a.sortie == b.sortie:
                        continue
                    k = tuple(sorted((a.id, b.id)))
                    if k in seen:
                        continue
                    seen.add(k)
                    ox = min(a.maxx, b.maxx) - max(a.minx, b.minx)
                    oy = min(a.maxy, b.maxy) - max(a.miny, b.miny)
                    if ox <= 0 or oy <= 0:
                        continue
                    area = ox * oy * M_PER_FTUS ** 2
                    if area >= min_overlap_m2:
                        pairs.append({
                            "a": a.id, "b": b.id, "sa": a.sortie, "sb": b.sortie,
                            "epsg": int(epsg), "overlap_m2": area,
                            "pa": a._asdict()["s3_path"], "pb": b._asdict()["s3_path"],
                            "x0": max(a.minx, b.minx), "x1": min(a.maxx, b.maxx),
                            "y0": max(a.miny, b.miny), "y1": min(a.maxy, b.maxy),
                        })
    p = pd.DataFrame(pairs)
    if p.empty:
        return p
    # spread the sample over distinct sortie combinations
    p["combo"] = p[["sa", "sb"]].apply(lambda r: " x ".join(sorted(r)), axis=1)
    rng = np.random.default_rng(seed)
    out = []
    for combo, g in p.groupby("combo"):
        take = min(len(g), max(1, max_pairs // max(p["combo"].nunique(), 1)))
        out.append(g.iloc[rng.choice(len(g), take, replace=False)])
    return pd.concat(out).head(max_pairs).reset_index(drop=True)


def fetch_xyz(s3_path):
    key = s3_path.replace("s3://arpa-i-insights/", "")
    buf = io.BytesIO()
    s3.download_fileobj("arpa-i-insights", key, buf)
    buf.seek(0)
    with laspy.open(buf) as f:
        las = f.read()
    x = np.array(las.x, float); y = np.array(las.y, float); z = np.array(las.z, float)
    del las; buf.close()
    return x, y, z


def grid_stats(x, y, z, x0, y0, nx, ny, cell_ft):
    """Per-cell count, 10th-percentile Z, and (p90-p10) spread, all in feet."""
    ix = np.floor((x - x0) / cell_ft).astype(np.int64)
    iy = np.floor((y - y0) / cell_ft).astype(np.int64)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[m], iy[m], z[m]
    flat = ix * ny + iy
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], z[order]
    bounds = np.searchsorted(fs, np.arange(nx * ny + 1))
    cnt = np.diff(bounds)
    g10 = np.full(nx * ny, np.nan)
    spread = np.full(nx * ny, np.nan)
    for c in np.nonzero(cnt >= MIN_PTS_PER_CELL)[0]:
        seg = zs[bounds[c]:bounds[c + 1]]
        p10, p90 = np.percentile(seg, [10, 90])
        g10[c] = p10
        spread[c] = p90 - p10
    return cnt, g10, spread


def compare(pair):
    cell_ft = CELL_M / M_PER_FTUS
    nx = max(int((pair["x1"] - pair["x0"]) / cell_ft), 1)
    ny = max(int((pair["y1"] - pair["y0"]) / cell_ft), 1)
    if nx * ny < 25:
        return None
    xa, ya, za = fetch_xyz(pair["pa"])
    xb, yb, zb = fetch_xyz(pair["pb"])
    ca, ga, sa = grid_stats(xa, ya, za, pair["x0"], pair["y0"], nx, ny, cell_ft)
    cb, gb, sb = grid_stats(xb, yb, zb, pair["x0"], pair["y0"], nx, ny, cell_ft)
    ok = (ca >= MIN_PTS_PER_CELL) & (cb >= MIN_PTS_PER_CELL)
    flatm = (sa * M_PER_FTUS < MAX_SPREAD_M) & (sb * M_PER_FTUS < MAX_SPREAD_M)
    sel = ok & flatm & np.isfinite(ga) & np.isfinite(gb)
    n = int(sel.sum())
    if n < 20:
        return {"a": pair["a"], "b": pair["b"], "combo": f'{pair["sa"]} x {pair["sb"]}',
                "cells": n, "note": "too few comparable cells"}
    dz = (ga[sel] - gb[sel]) * M_PER_FTUS
    med = float(np.median(dz))
    nmad = float(1.4826 * np.median(np.abs(dz - med)))
    return {
        "a": pair["a"], "b": pair["b"], "combo": f'{pair["sa"]} x {pair["sb"]}',
        "overlap_m2": round(pair["overlap_m2"]),
        "cells_compared": n,
        "cells_available": int(ok.sum()),
        "median_dz_m": round(med, 4),
        "nmad_dz_m": round(nmad, 4),
        "p95_abs_dz_m": round(float(np.percentile(np.abs(dz), 95)), 4),
        "rms_dz_m": round(float(np.sqrt(np.mean(dz ** 2))), 4),
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=glob.glob(
        "/workspace/data/lidar/v1/stac/index/*.parquet")[0])
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--min-overlap", type=float, default=4000.0,
                    help="minimum overlap area in m^2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260810)
    a = ap.parse_args()

    d = load_index(a.index)
    pairs = find_pairs(d, a.min_overlap, a.pairs, a.seed)
    if pairs.empty:
        print("no cross-sortie overlapping tile pairs found")
        return
    print(f"{len(pairs)} cross-sortie pairs across "
          f"{pairs['combo'].nunique()} sortie combinations", file=sys.stderr)
    print(pairs[["a", "b", "combo", "overlap_m2"]].to_string(index=False),
          file=sys.stderr)

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(compare, r): r["a"] for _, r in pairs.iterrows()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:
                print("  err", e, file=sys.stderr)
            print(f"  {i}/{len(futs)}", file=sys.stderr, flush=True)

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nPER-PAIR RELATIVE VERTICAL AGREEMENT (flat hard ground, 2 m cells)")
    print(r.to_string(index=False))

    good = r[r["note"] == ""]
    if len(good):
        print("\nPOOLED")
        print(f"  pairs compared            : {len(good)}")
        print(f"  cells compared            : {int(good['cells_compared'].sum())}")
        print(f"  median |median dz|        : {good['median_dz_m'].abs().median():.3f} m")
        print(f"  max    |median dz|        : {good['median_dz_m'].abs().max():.3f} m")
        print(f"  median NMAD of dz         : {good['nmad_dz_m'].median():.3f} m")
        print(f"  median RMS of dz          : {good['rms_dz_m'].median():.3f} m")
        print(f"  median p95 |dz|           : {good['p95_abs_dz_m'].median():.3f} m")
        print("\nCompare against the reported ABSOLUTE errors: 1.21 m X, 1.21 m Y, "
              "0.36 m Z, 1.71 m horizontal radial.")


if __name__ == "__main__":
    main()
