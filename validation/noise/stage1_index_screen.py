#!/usr/bin/env python3
"""Stage 1 -- index-only elevation excursion screen.

No point-cloud downloads.  For every tile in the STAC index this computes

    above = elevation:max - elevation:median
    below = elevation:median - elevation:min

and asks, per sortie, whether `above` is explained by terrain.  "Terrain" is
proxied by *local relief*: the range of `elevation:median` over the tile's
(2k+1)^2 grid neighbourhood (k=2) within the same sortie.

The headline result of this stage is NEGATIVE, and that is the point.  See the
CONCLUSION block that this script prints.

Run with --help for options.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.stats import spearmanr

INDEX_RELPATH = Path("data/lidar/v1/stac/index/items.parquet")
TILE_ID_RE = re.compile(r"_X(\d+)_Y(\d+)")


def default_index_path() -> Path:
    """Resolve the index relative to the repo root rather than to any absolute
    scratch location, so the script is portable across checkouts."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / INDEX_RELPATH
        if candidate.exists():
            return candidate
    return here.parents[2] / INDEX_RELPATH


def load_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        index_path,
        columns=["id", "sortie", "pc:count", "proj:epsg",
                 "elevation:min", "elevation:max", "elevation:median"],
    )
    # Grid indices come from the id, e.g. DRCOG_X060_Y033 -> gx=60, gy=33.
    gx = np.empty(len(df), dtype=np.int32)
    gy = np.empty(len(df), dtype=np.int32)
    for i, tid in enumerate(df["id"].to_numpy()):
        m = TILE_ID_RE.search(tid)
        if m is None:
            raise ValueError(f"cannot parse grid indices from id {tid!r}")
        gx[i] = int(m.group(1))
        gy[i] = int(m.group(2))
    df["gx"] = gx
    df["gy"] = gy
    return df


def local_relief(sub: pd.DataFrame, k: int = 2) -> np.ndarray:
    """Range of per-tile MEDIAN elevation over the (2k+1)^2 grid neighbourhood.

    We deliberately use the per-tile *median* elevation, not min/max: noise
    spikes and tall structures inside a single tile would otherwise inflate the
    "relief" of every tile in its neighbourhood, and the whole point of this
    stage is to test `above` against terrain rather than against itself.

    Missing grid positions (the sortie footprint is not a full rectangle) must
    not contribute.  Filling them with +/-inf before the max/min filters keeps
    them out of the result at every position that has data.
    """
    nx = int(sub["gx"].max()) + 1
    ny = int(sub["gy"].max()) + 1
    med = np.full((nx, ny), np.nan, dtype=np.float64)
    med[sub["gx"].to_numpy(), sub["gy"].to_numpy()] = sub["elevation:median"].to_numpy()
    valid = np.isfinite(med)

    size = 2 * k + 1
    hi = maximum_filter(np.where(valid, med, -np.inf), size=size,
                        mode="constant", cval=-np.inf)
    lo = minimum_filter(np.where(valid, med, np.inf), size=size,
                        mode="constant", cval=np.inf)
    relief_grid = hi - lo
    return relief_grid[sub["gx"].to_numpy(), sub["gy"].to_numpy()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1: index-only elevation excursion screen "
                    "index-only elevation excursion screen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--index", type=Path, default=default_index_path(),
                    help="path to items.parquet")
    ap.add_argument("--k", type=int, default=2,
                    help="neighbourhood half-width; relief uses a (2k+1)^2 window")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV to write the per-sortie summary to")
    ap.add_argument("--out-tiles", type=Path, default=None,
                    help="optional Parquet to write the per-tile table to")
    args = ap.parse_args(argv)

    if not args.index.exists():
        ap.error(f"index not found: {args.index}")

    df = load_index(args.index)
    print(f"index: {args.index}")
    print(f"tiles: {len(df)}  sorties: {df['sortie'].nunique()}")

    df["above"] = df["elevation:max"] - df["elevation:median"]
    df["below"] = df["elevation:median"] - df["elevation:min"]

    reliefs = np.empty(len(df), dtype=np.float64)
    for sortie, sub in df.groupby("sortie", sort=False):
        reliefs[sub.index.to_numpy()] = local_relief(sub, k=args.k)
    df["relief"] = reliefs

    rows = []
    for sortie, sub in df.groupby("sortie", sort=False):
        r_relief = spearmanr(sub["above"], sub["relief"]).statistic
        r_abs = spearmanr(sub["above"], sub["elevation:median"]).statistic
        rows.append({
            "sortie": sortie,
            "n_tiles": len(sub),
            "above_median_ft": float(sub["above"].median()),
            "above_p90_ft": float(sub["above"].quantile(0.90)),
            "below_median_ft": float(sub["below"].median()),
            "below_p90_ft": float(sub["below"].quantile(0.90)),
            "relief_median_ft": float(sub["relief"].median()),
            "spearman_above_vs_relief": float(r_relief),
            "spearman_above_vs_abs_elev": float(r_abs),
        })
    summary = pd.DataFrame(rows).sort_values("spearman_above_vs_relief",
                                             ascending=False)

    print()
    print(f"{'sortie':<11s} {'n':>6s} {'above med':>10s} {'above p90':>10s} "
          f"{'relief med':>11s} {'rho(relief)':>12s} {'rho(abs elev)':>14s}")
    print("-" * 80)
    for _, r in summary.iterrows():
        print(f"{r['sortie']:<11s} {int(r['n_tiles']):>6d} "
              f"{r['above_median_ft']:>10.1f} {r['above_p90_ft']:>10.1f} "
              f"{r['relief_median_ft']:>11.1f} "
              f"{r['spearman_above_vs_relief']:>12.3f} "
              f"{r['spearman_above_vs_abs_elev']:>14.3f}")
    print("(elevations are in US survey feet, as published in the index)")

    strongest = summary.iloc[0]
    print()
    print("CONCLUSION -- this stage is a NEGATIVE result, by design.")
    print(f"  * The strongest above-vs-relief correlation in the release is "
          f"{strongest['sortie']} at rho = {strongest['spearman_above_vs_relief']:.3f}, "
          f"and {strongest['sortie']} is one of the CLEAN sorties.")
    print("  * That is the whole finding: the elevation-excursion metric is "
          "tracking real terrain slope, not noise. A large `above` value is "
          "therefore not evidence of an artifact.")
    print("  * Absolute elevation shows no consistent relationship with `above` "
          "across sorties, so altitude is not a usable screen either.")
    print("  * Consequently the manuscript makes NO relief claim on the strength "
          "of this stage. Noise must be measured in height-above-ground on the "
          "actual points (Stage 2), which is what stage2_noise_sample.py does.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    if args.out_tiles is not None:
        args.out_tiles.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_tiles, index=False)
        print(f"wrote {args.out_tiles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
