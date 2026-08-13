"""Summarize inter-sortie vertical offsets, per sortie pair and per sortie.

Reads the per-pair measurements produced by `characterize_offsets.py` (JSONL) and needs no
network access, so it is cheap to re-run as more pairs are measured.

Two views, and the distinction matters:

* **Per sortie pair** is the measurement. Each row is a directly observed quantity: the
  median vertical difference between two sorties over ground they both observed.

* **Per sortie** is a summary of everything that sortie disagrees with. There is no
  absolute per-sortie elevation error obtainable this way. Overlaps constrain only
  *differences*, so any per-sortie number carries one arbitrary constant per connected
  component of the overlap graph, and a sortie that overlaps nothing is unconstrained
  entirely. The fitted-bias column is included to show how poorly a single bias per sortie
  explains the observations, not as a correction to apply.
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def load(path):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame([r for r in rows if r.get("ok")])
    if d.empty:
        sys.exit(f"no usable measurements in {path}")
    return d


def components(edges, nodes):
    par = {n: n for n in nodes}
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            par[ri] = rj
    out = defaultdict(list)
    for n in nodes:
        out[find(n)].append(n)
    return [sorted(v) for v in out.values()]


def fit_biases(pairs):
    """Least-squares per-sortie bias within each component, gauged to sum to zero."""
    obs = [(r.lo, r.hi, r.dz_median, r.pairs) for r in pairs.itertuples()]
    nodes = sorted({o[0] for o in obs} | {o[1] for o in obs})
    comps = components([(o[0], o[1]) for o in obs], nodes)
    bias, meta = {}, {}
    for cs in comps:
        sub = [o for o in obs if o[0] in cs and o[1] in cs]
        idx = {s: k for k, s in enumerate(cs)}
        A = np.zeros((len(sub) + 1, len(cs))); y = np.zeros(len(sub) + 1); w = np.ones(len(sub) + 1)
        for k, (i, j, dz, n) in enumerate(sub):
            A[k, idx[i]] = 1; A[k, idx[j]] = -1; y[k] = dz; w[k] = np.sqrt(n)
        A[-1, :] = 1; w[-1] = np.sqrt(max(len(sub), 1)) * 10
        b, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        resid = A[:-1] @ b - y[:-1] if sub else np.array([])
        dof = len(sub) - (len(cs) - 1)
        rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("nan")
        scale = float(np.median(np.abs([o[2] for o in sub]))) if sub else float("nan")
        for s in cs:
            bias[s] = float(b[idx[s]])
            meta[s] = dict(component=", ".join(cs), n_nodes=len(cs), n_edges=len(sub),
                           redundancy=dof, rms_resid=rms, offset_scale=scale)
    return bias, meta


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=os.path.join(here, "out", "offsets.jsonl"))
    ap.add_argument("--index", default=os.path.join(here, "..", "..", "data", "lidar", "v1",
                                                    "stac", "index", "items.parquet"))
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()

    d = load(a.pairs)
    all_sorties = sorted(pd.read_parquet(a.index, columns=["sortie"])["sortie"].unique())

    # ---- per sortie pair -------------------------------------------------------
    per = d.groupby("combo").agg(
        lo=("lo", "first"), hi=("hi", "first"),
        pairs=("dz_m", "size"), cells=("cells", "sum"),
        dz_median=("dz_m", "median"),
        dz_min=("dz_m", "min"), dz_max=("dz_m", "max"),
        dz_iqr=("dz_m", lambda x: float(x.quantile(.75) - x.quantile(.25))),
        shape_nmad=("nmad_m", "median"),
    ).reset_index()
    per["spread"] = per["dz_max"] - per["dz_min"]
    # standard error of a median over n cells, from the scatter about it
    per["se_mm"] = 1000 * per["shape_nmad"] / np.sqrt(per["cells"])

    print("=" * 100)
    print("ELEVATION DIFFERENCE BY SORTIE PAIR  (dz = first sortie minus second, metres)")
    print("=" * 100)
    show = per[["lo", "hi", "pairs", "cells", "dz_median", "dz_min", "dz_max",
                "spread", "dz_iqr", "shape_nmad", "se_mm"]].copy()
    show.columns = ["sortie A", "sortie B", "pairs", "cells", "dz median",
                    "dz min", "dz max", "spread", "IQR", "shape NMAD", "SE (mm)"]
    print(show.sort_values("dz median").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  spread = max-min across sampled pairs; IQR is robust to a single bad pair.")
    print("  A large IQR means the offset genuinely varies with location within that pair.")
    print("  shape NMAD is the scatter about the offset: agreement in surface shape.")

    # ---- per sortie -----------------------------------------------------------
    bias, meta = fit_biases(per)
    rows = []
    for s in all_sorties:
        mine = per[(per.lo == s) | (per.hi == s)]
        # express every observation as this sortie minus its partner
        obs = []
        for r in mine.itertuples():
            other = r.hi if r.lo == s else r.lo
            dz = r.dz_median if r.lo == s else -r.dz_median
            obs.append((other, dz))
        if not obs:
            rows.append(dict(sortie=s, partners=0, pairs=0, dz_median=np.nan,
                             abs_median=np.nan, dz_min=np.nan, dz_max=np.nan,
                             worst_partner="", shape_nmad=np.nan, fitted_bias=np.nan,
                             component="(no overlap)", redundancy=np.nan))
            continue
        vals = np.array([o[1] for o in obs])
        worst = max(obs, key=lambda o: abs(o[1]))
        m = meta.get(s, {})
        rows.append(dict(
            sortie=s, partners=len(obs), pairs=int(mine["pairs"].sum()),
            dz_median=float(np.median(vals)), abs_median=float(np.median(np.abs(vals))),
            dz_min=float(vals.min()), dz_max=float(vals.max()),
            worst_partner=f"{worst[0]} {worst[1]:+.2f}",
            shape_nmad=float(mine["shape_nmad"].median()),
            fitted_bias=bias.get(s, np.nan),
            component=m.get("component", ""), redundancy=m.get("redundancy", np.nan)))
    ps = pd.DataFrame(rows)

    print()
    print("=" * 100)
    print("ELEVATION DIFFERENCE BY SORTIE  (metres; each value is this sortie minus a partner)")
    print("=" * 100)
    disp = ps[["sortie", "partners", "pairs", "dz_median", "abs_median", "dz_min", "dz_max",
               "worst_partner", "shape_nmad", "fitted_bias"]].copy()
    disp.columns = ["sortie", "partners", "pairs", "dz median", "|dz| median", "dz min",
                    "dz max", "largest disagreement", "shape NMAD", "fitted bias"]
    def fmt(v):
        return "  --  " if pd.isna(v) else f"{v:.3f}"
    print(disp.to_string(index=False, float_format=fmt, na_rep="  --  "))

    none = ps[ps.partners == 0]["sortie"].tolist()
    print(f"\n  Sorties overlapping no other sortie, so not assessable by this method "
          f"({len(none)}): {', '.join(none) if none else 'none'}")

    print("\n  Overlap-graph components, and whether a per-sortie bias is testable:")
    seen = set()
    for s in all_sorties:
        c = ps.loc[ps.sortie == s, "component"].iloc[0]
        if not c or c in seen or c == "(no overlap)":
            continue
        seen.add(c)
        m = meta[s]
        verdict = ("no redundancy (tree): a per-sortie bias reproduces the observations by "
                   "construction and cannot be tested"
                   if m["redundancy"] <= 0 else
                   f"redundancy {m['redundancy']}: RMS residual {m['rms_resid']:.3f} m against a "
                   f"typical offset of {m['offset_scale']:.3f} m "
                   f"({100*m['rms_resid']/m['offset_scale']:.0f}%)")
        print(f"    {{{c}}}\n      {verdict}")
    print("\n  The fitted-bias column is NOT a correction. It carries one arbitrary constant")
    print("  per component, and where redundancy exists the fit is poor, so the offsets are")
    print("  not reducible to one value per sortie. Estimate the offset locally instead.")

    if a.out_prefix:
        per.to_csv(f"{a.out_prefix}_by_pair.csv", index=False)
        ps.to_csv(f"{a.out_prefix}_by_sortie.csv", index=False)
        print(f"\nwrote {a.out_prefix}_by_pair.csv and {a.out_prefix}_by_sortie.csv")


if __name__ == "__main__":
    main()
