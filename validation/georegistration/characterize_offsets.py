"""Full characterization of inter-sortie vertical offsets in INSIGHTS-LiDAR.

Where two sorties overlap, the same ground is observed twice, so the two surfaces can be
differenced directly. A preliminary screen found metre-scale vertical offsets with
centimetre-scale shape agreement; this quantifies them across every overlapping sortie
pair in the release and tests whether they are explained by a single bias per sortie.

Design
------
* Candidate pairs come from `proj:bbox` overlap. That is a superset for rotated grids
  (the box circumscribes the occupied region), so genuine overlap is confirmed by
  requiring enough co-occupied cells; spurious pairs drop out.
* Comparison surface: 2 m grid over the overlap, per-cell ground proxy = 10th percentile
  of Z. A cell is used only if both sorties have >= 30 points in it and the within-cell
  p90-p10 spread is under 0.5 m in both, restricting to flat, hard, fully sampled ground.
* Per pair we record the median difference (the offset), its NMAD (shape agreement),
  and the planar tilt, so a constant offset can be distinguished from a warp.
* Pairs within a combination are spread spatially, so a combination's offset is not
  estimated from a single location.
* EXCLUSION: the western third of I80P2 (gx < 27; gx increases eastward) is omitted.
  That part of the sortie carries dense noise that would corrupt a ground estimate.

Network test
------------
Overlaps form a graph on sorties. Within each connected component we solve
    d_ij = b_i - b_j
by least squares for a per-sortie bias b (gauge: biases sum to zero within the
component). Components with more edges than nodes-1 are over-determined, so the
residuals test the hypothesis that the offsets are a per-sortie property. Small
residuals mean a per-sortie vertical correction exists; large residuals mean the offsets
are pair- or location-specific and no such correction exists.
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
MIN_CELLS = 60          # per pair, to accept an offset estimate

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32))


def load_index(path):
    d = pd.read_parquet(path, columns=["id", "sortie", "proj:bbox", "proj:epsg", "s3_path"])
    # rename to valid identifiers: itertuples() below renames "proj:epsg" positionally,
    # which is fragile to any change in column order
    d = d.rename(columns={"proj:bbox": "pbbox", "proj:epsg": "epsg"})
    bb = np.array([json.loads(x) for x in d["pbbox"]], dtype=float)
    d["minx"], d["miny"], d["maxx"], d["maxy"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    xy = d["id"].str.extract(r"_X(\d+)_Y(\d+)")
    d["gx"] = xy[0].astype(int)
    d["gy"] = xy[1].astype(int)
    return d


def candidate_pairs(d, min_overlap_m2):
    step = 500 / M_PER_FTUS
    d = d.assign(bx=(d["minx"] / step).astype(int), by=(d["miny"] / step).astype(int))
    from collections import defaultdict
    bk = defaultdict(list)
    for r in d.itertuples():
        for dx in (-1, 0):
            for dy in (-1, 0):
                bk[(r.epsg, r.bx + dx, r.by + dy)].append(r)
    seen, rows = set(), []
    for _, rs in bk.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a.sortie == b.sortie:
                    continue
                k = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if k in seen:
                    continue
                seen.add(k)
                ox = min(a.maxx, b.maxx) - max(a.minx, b.minx)
                oy = min(a.maxy, b.maxy) - max(a.miny, b.miny)
                if ox <= 0 or oy <= 0:
                    continue
                area = ox * oy * M_PER_FTUS ** 2
                if area < min_overlap_m2:
                    continue
                lo, hi = sorted((a.sortie, b.sortie))
                rows.append({
                    "combo": f"{lo} x {hi}", "lo": lo, "hi": hi,
                    "a": a.id, "b": b.id, "sa": a.sortie, "sb": b.sortie,
                    "pa": a.s3_path, "pb": b.s3_path, "overlap_m2": area,
                    "x0": max(a.minx, b.minx), "x1": min(a.maxx, b.maxx),
                    "y0": max(a.miny, b.miny), "y1": min(a.maxy, b.maxy),
                })
    return pd.DataFrame(rows)


def spread_sample(p, per_combo, seed):
    """Take per_combo pairs from each combination, spread along the overlap corridor."""
    out = []
    for combo, g in p.groupby("combo"):
        g = g.assign(pos=g["x0"] + g["y0"]).sort_values("pos")
        if len(g) <= per_combo:
            out.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, per_combo).round().astype(int)
            out.append(g.iloc[np.unique(idx)])
    return pd.concat(out).reset_index(drop=True)


def fetch(s3_path):
    key = s3_path.replace("s3://arpa-i-insights/", "")
    buf = io.BytesIO()
    s3.download_fileobj("arpa-i-insights", key, buf)
    buf.seek(0)
    with laspy.open(buf) as f:
        las = f.read()
    x = np.array(las.x, float); y = np.array(las.y, float); z = np.array(las.z, float)
    del las; buf.close()
    return x, y, z


def cellstats(x, y, z, x0, y0, nx, ny, cell_ft):
    ix = np.floor((x - x0) / cell_ft).astype(np.int64)
    iy = np.floor((y - y0) / cell_ft).astype(np.int64)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[m], iy[m], z[m]
    f = ix * ny + iy
    o = np.argsort(f, kind="stable")
    fs, zs = f[o], z[o]
    b = np.searchsorted(fs, np.arange(nx * ny + 1))
    cnt = np.diff(b)
    g = np.full(nx * ny, np.nan)
    sp = np.full(nx * ny, np.nan)
    for c in np.nonzero(cnt >= MIN_PTS_PER_CELL)[0]:
        seg = zs[b[c]:b[c + 1]]
        p10, p90 = np.percentile(seg, [10, 90])
        g[c] = p10
        sp[c] = p90 - p10
    return cnt, g, sp


def measure(r):
    cell_ft = CELL_M / M_PER_FTUS
    nx = max(int((r["x1"] - r["x0"]) / cell_ft), 1)
    ny = max(int((r["y1"] - r["y0"]) / cell_ft), 1)
    if nx * ny < MIN_CELLS:
        return None
    xa, ya, za = fetch(r["pa"])
    xb, yb, zb = fetch(r["pb"])
    ca, ga, sa = cellstats(xa, ya, za, r["x0"], r["y0"], nx, ny, cell_ft)
    cb, gb, sb = cellstats(xb, yb, zb, r["x0"], r["y0"], nx, ny, cell_ft)
    sel = ((ca >= MIN_PTS_PER_CELL) & (cb >= MIN_PTS_PER_CELL)
           & (sa * M_PER_FTUS < MAX_SPREAD_M) & (sb * M_PER_FTUS < MAX_SPREAD_M)
           & np.isfinite(ga) & np.isfinite(gb))
    n = int(sel.sum())
    if n < MIN_CELLS:
        return {"combo": r["combo"], "a": r["a"], "b": r["b"], "cells": n, "ok": False}
    dz = (ga[sel] - gb[sel]) * M_PER_FTUS
    med = float(np.median(dz))
    nmad = float(1.4826 * np.median(np.abs(dz - med)))
    ii = np.nonzero(sel)[0]
    cx, cy = (ii // ny) * CELL_M, (ii % ny) * CELL_M
    tilt = np.nan
    if len(np.unique(cx)) > 3 and len(np.unique(cy)) > 3:
        A = np.column_stack([cx, cy, np.ones_like(cx)])
        coef, *_ = np.linalg.lstsq(A, dz, rcond=None)
        tilt = float(np.hypot(coef[0], coef[1]) * 1000)      # mm per m
    # SIGN CONVENTION. Which tile is "a" depends on iteration order, so within one
    # combination "a" is not always the same sortie. Normalize every observation to
    # (alphabetically first sortie) minus (second), or the signs would cancel when
    # aggregated.
    if r["sa"] != r["lo"]:
        med = -med
    return {"combo": r["combo"], "lo": r["lo"], "hi": r["hi"],
            "a": r["a"], "b": r["b"], "sa": r["sa"], "sb": r["sb"],
            "cells": n, "ok": True, "dz_m": round(med, 4), "nmad_m": round(nmad, 4),
            "tilt_mm_per_m": None if np.isnan(tilt) else round(tilt, 2),
            "overlap_m2": round(r["overlap_m2"])}


def components(edges, nodes):
    parent = {n: n for n in nodes}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    comp = {}
    for n in nodes:
        comp.setdefault(find(n), []).append(n)
    return list(comp.values())


def network(per_combo):
    """Least-squares per-sortie bias within each connected component."""
    obs = []
    for _, r in per_combo.iterrows():
        obs.append((r["sa_ref"], r["sb_ref"], r["dz_median"], r["n_pairs"]))
    nodes = sorted({o[0] for o in obs} | {o[1] for o in obs})
    comps = components([(o[0], o[1]) for o in obs], nodes)
    print("\n" + "=" * 78)
    print("NETWORK ADJUSTMENT: is the offset a per-sortie bias?")
    print("=" * 78)
    for comp in sorted(comps, key=len, reverse=True):
        sub = [o for o in obs if o[0] in comp and o[1] in comp]
        if not sub:
            continue
        idx = {s: k for k, s in enumerate(sorted(comp))}
        A = np.zeros((len(sub) + 1, len(comp)))
        y = np.zeros(len(sub) + 1)
        w = np.ones(len(sub) + 1)
        for k, (i, j, d, npair) in enumerate(sub):
            A[k, idx[i]] = 1.0
            A[k, idx[j]] = -1.0
            y[k] = d
            w[k] = np.sqrt(npair)
        A[-1, :] = 1.0            # gauge: biases sum to zero
        y[-1] = 0.0
        w[-1] = np.sqrt(len(sub)) * 10
        Aw = A * w[:, None]; yw = y * w
        b, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        resid = A[:-1] @ b - y[:-1]
        dof = len(sub) - (len(comp) - 1)
        print(f"\ncomponent: {', '.join(sorted(comp))}")
        print(f"  observations {len(sub)}, unknowns {len(comp)-1}, redundancy {dof}")
        print("  fitted per-sortie vertical bias (m, relative to component mean):")
        for s in sorted(comp):
            print(f"    {s:11s} {b[idx[s]]:+8.3f}")
        if dof > 0:
            print("  residuals of the pairwise observations (m):")
            for k, (i, j, d, _) in enumerate(sub):
                print(f"    {i:11s} - {j:11s}  observed {d:+7.3f}   "
                      f"fitted {b[idx[i]]-b[idx[j]]:+7.3f}   residual {resid[k]:+7.3f}")
            rms = float(np.sqrt(np.mean(resid ** 2)))
            mx = float(np.max(np.abs(resid)))
            scale = float(np.median(np.abs([o[2] for o in sub])))
            print(f"  RMS residual: {rms:.3f} m   max |residual|: {mx:.3f} m")
            print(f"  median |observed offset|: {scale:.3f} m")
            # Verdict must follow the numbers. Residuals are only evidence for a
            # per-sortie bias if they are small compared with the offsets being
            # explained; if they are comparable, the offsets are pair- or
            # location-specific and no per-sortie correction exists.
            ratio = rms / scale if scale > 0 else float("inf")
            if ratio < 0.25:
                print(f"  VERDICT: residuals are {ratio:.0%} of the typical offset. The "
                      "offsets are\n    consistent with one bias per sortie, so a "
                      "per-sortie vertical correction exists.")
            elif ratio < 0.5:
                print(f"  VERDICT: residuals are {ratio:.0%} of the typical offset. A "
                      "per-sortie bias\n    explains part of the signal but leaves "
                      "substantial pair-specific residual.")
            else:
                print(f"  VERDICT: residuals are {ratio:.0%} of the typical offset, i.e. "
                      "comparable to\n    the offsets themselves. A single bias per sortie "
                      "does NOT explain them, so\n    no per-sortie correction exists; the "
                      "offset must be estimated locally.")
        else:
            print("  no redundancy (tree); offsets are reproduced exactly by construction "
                  "and\n    consistency cannot be tested within this component.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=glob.glob(
        "/workspace/data/lidar/v1/stac/index/*.parquet")[0])
    ap.add_argument("--per-combo", type=int, default=10)
    ap.add_argument("--min-overlap", type=float, default=6000.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--out", default="offsets.jsonl")
    a = ap.parse_args()

    d = load_index(a.index)
    n0 = len(d)
    d = d[~((d["sortie"] == "I80P2") & (d["gx"] < 27))]
    print(f"excluded {n0-len(d)} tiles: western third of I80P2 (dense noise)",
          file=sys.stderr)

    cand = candidate_pairs(d, a.min_overlap)
    samp = spread_sample(cand, a.per_combo, a.seed)
    print(f"{len(cand)} candidate pairs -> measuring {len(samp)} across "
          f"{samp['combo'].nunique()} combinations", file=sys.stderr)

    rows = []
    with open(a.out, "w") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(measure, r): i for i, r in samp.iterrows()}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                if r:
                    rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
            except Exception as e:
                print("  err", e, file=sys.stderr)
            if k % 20 == 0:
                print(f"  {k}/{len(futs)}", file=sys.stderr, flush=True)

    r = pd.DataFrame(rows)
    good = r[r["ok"]].copy() if "ok" in r else r
    pd.set_option("display.width", 250)
    print(f"\nmeasured {len(good)} usable pairs of {len(r)} attempted")

    print("\n" + "=" * 78)
    print("PER-COMBINATION VERTICAL OFFSET")
    print("=" * 78)
    print("dz_median is the offset; dz_spread is its range across the sampled pairs, so a")
    print("large spread means the offset is not constant even within one sortie pair.")
    print("shape_nmad is the scatter about the offset: agreement in surface shape.")
    agg = good.groupby("combo").agg(
        pairs=("dz_m", "size"),
        cells=("cells", "sum"),
        dz_median=("dz_m", "median"),
        dz_spread=("dz_m", lambda x: float(x.max() - x.min())),
        dz_iqr=("dz_m", lambda x: float(x.quantile(.75) - x.quantile(.25))),
        shape_nmad=("nmad_m", "median"),
        tilt=("tilt_mm_per_m", "median"),
    )
    print(agg.round(3).to_string())

    ref = good.groupby("combo").first()[["lo", "hi"]]
    per = agg.join(ref).reset_index()
    per["sa_ref"], per["sb_ref"] = per["lo"], per["hi"]
    per["n_pairs"] = per["pairs"]
    network(per)

    good.to_csv("offsets_pairs.csv", index=False)
    agg.to_csv("offsets_per_combo.csv")
    print("\nwrote offsets_pairs.csv, offsets_per_combo.csv, " + a.out)


if __name__ == "__main__":
    main()
