#!/usr/bin/env python3
"""Stage 2 reporting -- per-sortie noise table.

Reads the JSONL written by stage2_noise_sample.py and derives the per-sortie
table.  No network access, no recomputation of points: everything here is
arithmetic on the per-tile cell statistics.

Rates are reconstructed from the cell statistics exactly as section 3.6
prescribes:

    affected_cells = cell_frac * n_cells_occupied
    noise_points   = pts_per_affected_cell * affected_cells
    rate           = noise_points / n_points        (reported in ppm)

Run with --help for options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# A tile counts as AFFECTED when its isolated high-noise rate exceeds 10 ppm.
AFFECTED_RATE = 1e-5

SORTIE_ORDER = ["DRCOG", "FrontRange", "I15South", "I25N", "I25S2", "I70A",
                "I70BC", "I70D", "I70E", "I70F", "I70G", "I70H", "I70I",
                "I80East", "I80P1", "I80P2", "SLC"]


def default_tiles_path() -> Path:
    return Path(__file__).resolve().parents[1] / "out" / "stage2_tiles.jsonl"


def load_tiles(path: Path) -> pd.DataFrame:
    recs = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"no records in {path}")
    df = pd.DataFrame(recs).drop_duplicates(subset="id", keep="last")
    return df


def derive_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for pre in ("hi", "lo"):
        affected_cells = df[f"{pre}_cell_frac"] * df["n_cells_occupied"]
        noise_points = df[f"{pre}_pts_per_affected_cell"] * affected_cells
        df[f"{pre}_rate"] = noise_points / df["n_points"]
        df[f"{pre}_rate_ppm"] = 1e6 * df[f"{pre}_rate"]
    return df


def fmt(v, spec: str = "{:.1f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return spec.format(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-sortie noise table from the Stage 2 JSONL "
                    "per-sortie noise table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--tiles", type=Path, default=default_tiles_path(),
                    help="Stage 2 JSONL produced by stage2_noise_sample.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV output path for the per-sortie table")
    ap.add_argument("--out-tiles", type=Path, default=None,
                    help="optional CSV output path for the per-tile table "
                         "with derived rates")
    ap.add_argument("--affected-rate", type=float, default=AFFECTED_RATE,
                    help="isolated high-noise rate above which a tile counts as "
                         "affected")
    args = ap.parse_args(argv)

    if not args.tiles.exists():
        ap.error(f"tiles file not found: {args.tiles}")

    df = derive_rates(load_tiles(args.tiles))
    print(f"tiles: {args.tiles}")
    print(f"records: {len(df)} across {df['sortie'].nunique()} sorties")
    n_per = df.groupby("sortie").size()
    print(f"tiles per sortie: min {n_per.min()} max {n_per.max()}")
    if n_per.max() < 30:
        print("NOTE: this is a reduced sample (n<30). The section 3.7 table was "
              "measured at n=30; medians and p90s from a smaller sample will not "
              "match it and should not be tuned to.")
    print()

    rows = []
    order = [s for s in SORTIE_ORDER if s in set(df["sortie"])]
    order += sorted(set(df["sortie"]) - set(order))
    for sortie in order:
        sub = df[df["sortie"] == sortie]
        # Character metrics are aggregated over AFFECTED tiles only.  A median
        # over all sampled tiles reads as zero for sorties whose noise is
        # localised (SLC, DRCOG, I80P2), which hides the character entirely.
        aff = sub[sub["hi_rate"] > args.affected_rate]
        band_p50 = aff["iso20_p50"].dropna()
        band_iqr = aff["iso20_iqr"].dropna()
        rows.append({
            "sortie": sortie,
            "n_tiles": len(sub),
            "hi_med_ppm": sub["hi_rate_ppm"].median(),
            "hi_p90_ppm": sub["hi_rate_ppm"].quantile(0.90),
            "lo_med_ppm": sub["lo_rate_ppm"].median(),
            "lo_p90_ppm": sub["lo_rate_ppm"].quantile(0.90),
            "pct_hi": 100 * (sub["hi_rate"] > args.affected_rate).mean(),
            "pct_lo": 100 * (sub["lo_rate"] > args.affected_rate).mean(),
            "n_affected": len(aff),
            "el20_extent_pct": 100 * aff["el20_cell_frac"].median() if len(aff) else np.nan,
            "el20_dens_m2": aff["el20_areal_density_m2"].median() if len(aff) else np.nan,
            "el20_density_ratio": aff["el20_density_ratio"].median() if len(aff) else np.nan,
            "band_p50_m": band_p50.median() if len(band_p50) else np.nan,
            "band_iqr_m": band_iqr.median() if len(band_iqr) else np.nan,
        })
    tab = pd.DataFrame(rows)

    hdr = (f"{'sortie':<11s} {'n':>3s} {'hi med':>8s} {'hi p90':>8s} "
           f"{'lo med':>8s} {'lo p90':>8s} {'%hi':>5s} {'%lo':>5s} "
           f"{'extent':>7s} {'dens':>6s} {'ratio':>7s} {'band':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in tab.iterrows():
        band = ("—" if not np.isfinite(r["band_p50_m"])
                else f"{r['band_p50_m']:.0f} ({r['band_iqr_m']:.0f})")
        print(f"{r['sortie']:<11s} {int(r['n_tiles']):>3d} "
              f"{r['hi_med_ppm']:>8.1f} {r['hi_p90_ppm']:>8.0f} "
              f"{r['lo_med_ppm']:>8.1f} {r['lo_p90_ppm']:>8.0f} "
              f"{r['pct_hi']:>5.0f} {r['pct_lo']:>5.0f} "
              f"{fmt(r['el20_extent_pct']):>7s} {fmt(r['el20_dens_m2']):>6s} "
              f"{fmt(r['el20_density_ratio'], '{:.3f}'):>7s} {band:>12s}")
    print()
    print("columns: hi/lo med,p90 = isolated high/low noise rate in ppm over all "
          "sampled tiles;")
    print("         %hi,%lo = share of tiles above the "
          f"{1e6 * args.affected_rate:.0f} ppm threshold;")
    print("         extent = median % of occupied 5 m cells holding a return "
          "above 20 m AGL;")
    print("         dens   = median areal density of that elevated population "
          "(points/m^2);")
    print("         ratio  = dens as a fraction of the tile's own cell density;")
    print("         band   = median (IQR) height of the ISOLATED >20 m "
          "population, m.")
    print("         extent/dens/ratio/band are over AFFECTED tiles only.")
    print()
    print("Two apparent contradictions in this table are both correct:")
    print("  * a sortie can show %hi = 0 with a nonzero hi p90 -- the p90 simply "
          "lies below the 10 ppm affected threshold;")
    print("  * `dens` describes the ELEVATED population (no isolation filter) "
          "while `band` describes the ISOLATED one, so their point counts are "
          "not comparable.")
    print()
    print("KEY READING: `ratio` is the discriminator. An artifact layer deposits "
          "a few points per m^2 (ratio ~0.004-0.16); a real roof or canopy is "
          "sampled at close to the tile's own density (ratio ~1). See "
          "stage2_checks.py negative-control for the real-highrise side of that "
          "comparison.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    if args.out_tiles is not None:
        args.out_tiles.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_tiles, index=False)
        print(f"wrote {args.out_tiles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
