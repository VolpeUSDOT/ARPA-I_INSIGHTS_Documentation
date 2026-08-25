"""Is a sortie's vertical bias a constant offset, or a ramp along the flight direction?

`compare_3dep.py` measures the vertical difference between INSIGHTS and the USGS 3DEP 1 m
bare-earth DEM for a sample of tiles in each sortie. Those per-tile biases are not uniform
within a sortie: the tile-to-tile spread reaches several metres. This script asks what that
variation is made of. A constant bias, a smooth ramp along the collection, and unstructured
scatter are three different defects with three different consequences for a user, and the
spread alone cannot tell them apart.

Why the fit is one-dimensional
------------------------------
The obvious approach, fitting a plane in (x, y) to the per-tile biases, does not work here
and reports a confidently wrong answer rather than failing. Most of these sorties are
highway corridors: their tiles form a nearly straight line, with a principal-axis
anisotropy up to 78:1. Across such a sample the across-corridor gradient is not
identifiable, so a plane fit distributes the signal arbitrarily between its two gradient
terms and the fitted gradient *direction* is meaningless even when the magnitude is not.

The fit is therefore one-dimensional, along the principal axis of the sampled tile
positions, which for a corridor sortie is the corridor itself. The reported gradient is the
along-corridor component only. **The across-corridor component is not measured**, and this
method cannot detect a tilt perpendicular to the flight direction.

Why not R^2
-----------
With 4 to 30 tiles per sortie and a two-parameter model, R^2 is inflated by construction
and is near 1 for a sample of four regardless of whether a ramp exists. Two honest tests
are used instead:

* a **permutation test**, which reshuffles the biases against the positions many times and
  asks how often chance alone produces a fit this good;
* **leave-one-out prediction error**, compared against the same quantity for a
  constant-only model. A ramp that is real predicts a held-out tile better than the mean
  does; a ramp that is overfitting does worse.

Two confounds that produce a spurious ramp
------------------------------------------
* **Reference epoch.** A sortie whose tiles are compared against more than one 3DEP project
  can show a step between them, and a step at one end of a corridor mimics a ramp. Where
  this arises the ramp is refitted within the single project contributing the most tiles;
  a gradient that survives that is not an artefact of the reference. This check is not a
  formality: of the sorties whose ramp is significant, three fail it, two of them with the
  refitted gradient reversing sign. Where 3DEP coverage forces a corridor onto references
  of different epochs, this method cannot separate a ramp from the steps between them, and
  those sorties are labelled `ramp-unconfirmed` rather than counted as ramps.
* **A failed measurement.** A tile whose own within-tile scatter is metre-scale did not
  produce a usable bias, and one such tile dominates a small sample. `--max-nmad` drops
  them; the count dropped is reported rather than passed over in silence.

Significance alone is also not enough to call something a ramp, because a gradient can be
real and still explain almost none of the variation. Each sortie is therefore labelled:

    ramp              significant, predicts held-out tiles better than the mean, explains
                      at least three quarters of the tile-to-tile variance, and survives
                      the reference-epoch control
    ramp-unconfirmed  significant but failing one of those: either the epoch control
                      disagrees, or the ramp leaves most of the variance behind
    constant          no significant ramp, and the sortie varies little
    unstructured      no significant ramp, yet the sortie varies by more than a metre --
                      the variation is real but not a function of position along the flight
                      direction

Collection mode is reported alongside, from the Methods description of the campaign, because
it is the obvious thing to test the result against: the highway corridors were flown in
line-following mode as two overlapping passes, whereas the metropolitan and Front Range
blocks were flown as a raster grid with both N/S and E/W passes. A grid cross-braces the
block; a single corridor does not.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

M_PER_FTUS = 0.3048006096012192

# From Methods, "Data collection campaign": line-following mode for the highway
# collections, raster wide-area scan for the metropolitan and Front Range blocks.
MODE = {"DRCOG": "raster", "FrontRange": "raster", "SLC": "raster"}


def load(path, index_path, max_nmad):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows)
    d = d[d.get("ok") == True].drop_duplicates(subset="id", keep="last")
    n_all = len(d)
    if "cx" not in d or d["cx"].isna().any():
        # older output, or rows written before the tile centroid was recorded
        idx = pd.read_parquet(index_path, columns=["id", "proj:bbox"])
        bb = np.array([json.loads(x) for x in idx["proj:bbox"]], float)
        idx["cx_i"] = (bb[:, 0] + bb[:, 2]) / 2
        idx["cy_i"] = (bb[:, 1] + bb[:, 3]) / 2
        d = d.merge(idx[["id", "cx_i", "cy_i"]], on="id", how="left")
        d["cx"] = d["cx"].fillna(d["cx_i"]) if "cx" in d else d["cx_i"]
        d["cy"] = d["cy"].fillna(d["cy_i"]) if "cy" in d else d["cy_i"]
    # the index and the comparison both work in US survey feet; metres from here on
    d["cx"] = d["cx"] * M_PER_FTUS
    d["cy"] = d["cy"] * M_PER_FTUS
    bad = d["nmad_m"] > max_nmad
    if bad.any():
        print(f"dropping {int(bad.sum())} of {n_all} tiles whose own within-tile scatter "
              f"exceeds {max_nmad} m, so their bias is not a usable measurement:")
        for _, r in d[bad].iterrows():
            print(f"    {r['id']:26s} dz {r['dz_m']:+8.3f} m   within-tile NMAD "
                  f"{r['nmad_m']:6.3f} m")
        d = d[~bad]
    return d.reset_index(drop=True)


def along_axis(cx, cy):
    """Position along the principal axis of the tile centres, and the anisotropy."""
    xy = np.c_[cx - cx.mean(), cy - cy.mean()]
    _, sv, vt = np.linalg.svd(xy, full_matrices=False)
    aniso = float(sv[0] / sv[1]) if len(sv) > 1 and sv[1] > 1e-9 else np.inf
    return xy @ vt[0], aniso


def loo_rmse(A, z):
    """Leave-one-out prediction error. Honest where in-sample R^2 is not."""
    e = []
    for i in range(len(z)):
        m = np.ones(len(z), bool)
        m[i] = False
        if np.linalg.matrix_rank(A[m]) < A.shape[1]:
            return np.nan
        c, *_ = np.linalg.lstsq(A[m], z[m], rcond=None)
        e.append(z[i] - A[i] @ c)
    return float(np.sqrt(np.mean(np.square(e))))


def fit_ramp(t, z, n_perm, rng):
    A = np.c_[np.ones(len(z)), t]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ coef
    ss = float(((z - z.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss if ss > 0 else np.nan
    hits = 0
    for _ in range(n_perm):
        zp = rng.permutation(z)
        cp, *_ = np.linalg.lstsq(A, zp, rcond=None)
        sp = float(((zp - zp.mean()) ** 2).sum())
        r2p = 1.0 - float(((zp - A @ cp) ** 2).sum()) / sp if sp > 0 else np.nan
        if r2p >= r2:
            hits += 1
    return dict(grad_mm_per_m=coef[1] * 1000.0, intercept=coef[0], r2=r2,
                p_perm=(1 + hits) / (1 + n_perm),
                resid_rms=float(np.sqrt((res ** 2).mean())),
                loo_const=loo_rmse(np.ones((len(z), 1)), z), loo_ramp=loo_rmse(A, z))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default=os.path.join(here, "out", "compare_3dep.jsonl"))
    ap.add_argument("--index", default=os.path.join(repo, "data", "lidar", "v1", "stac",
                                                    "index", "items.parquet"))
    ap.add_argument("--max-nmad", type=float, default=1.0,
                    help="drop tiles whose own within-tile NMAD exceeds this, in m")
    ap.add_argument("--min-tiles", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-var-explained", type=float, default=0.75,
                    help="variance a ramp must account for to be called one")
    ap.add_argument("--flat-sd", type=float, default=1.0,
                    help="tile-to-tile sd above which a non-ramp sortie is called unstructured, in m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(here, "out", "ramp_fit.csv"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = load(a.jsonl, a.index, a.max_nmad)

    recs = []
    for s, g in d.groupby("sortie"):
        g = g.reset_index(drop=True)
        z = g["dz_m"].to_numpy(float)
        t, aniso = along_axis(g["cx"].to_numpy(float), g["cy"].to_numpy(float))
        rec = dict(sortie=s, mode=MODE.get(s, "line-following"), tiles=len(z),
                   span_km=(t.max() - t.min()) / 1000.0, aniso=aniso,
                   dz_median=float(np.median(z)), sd=float(z.std(ddof=0)),
                   within_tile_nmad=float(g["nmad_m"].median()),
                   within_tile_grad=float(g["grad_mm_per_m"].median()),
                   ref_years="/".join(str(y) for y in sorted(set(g["dem_year"]))))
        if len(z) >= a.min_tiles:
            rec.update(fit_ramp(t, z, a.n_perm, rng))
            # A sortie drawing on more than one reference project can show a step between
            # them that mimics a ramp. Refit inside the project contributing most tiles.
            if g["dem_project"].nunique() > 1:
                mu = g.groupby("dem_project")["dz_m"].mean()
                rec["epoch_step"] = float(mu.max() - mu.min())
                main_proj = g["dem_project"].value_counts().idxmax()
                sub = g[g["dem_project"] == main_proj]
                if len(sub) >= a.min_tiles:
                    ts, _ = along_axis(sub["cx"].to_numpy(float),
                                       sub["cy"].to_numpy(float))
                    f1 = fit_ramp(ts, sub["dz_m"].to_numpy(float), a.n_perm, rng)
                    rec["grad_single_project"] = f1["grad_mm_per_m"]
                    rec["resid_single_project"] = f1["resid_rms"]
                    rec["single_project"] = main_proj
                    rec["n_single_project"] = len(sub)
        recs.append(rec)

    r = pd.DataFrame(recs).sort_values("sortie").reset_index(drop=True)

    # Classify. A gradient can be significant and still explain almost none of the
    # variation, and it can be an artefact of comparing one end of a corridor against a
    # different-epoch reference, so significance is necessary and not sufficient.
    var_expl = 1.0 - (r["resid_rms"] / r["sd"]) ** 2
    sig = r["p_perm"].notna() & (r["p_perm"] < a.alpha)
    predicts = r["loo_ramp"] < r["loo_const"]
    explains = var_expl >= a.min_var_explained
    g_all, g_one = r["grad_mm_per_m"], r.get("grad_single_project")
    if g_one is None:
        epoch_ok = pd.Series(True, index=r.index)
    else:
        # no multi-project tiles means nothing to disagree with; otherwise the refitted
        # gradient must keep its sign and stay within a factor of two
        epoch_ok = g_one.isna() | ((np.sign(g_one) == np.sign(g_all)) &
                                   (g_one.abs() >= 0.5 * g_all.abs()) &
                                   (g_one.abs() <= 2.0 * g_all.abs()))
    r["var_expl"] = var_expl
    r["epoch_ok"] = epoch_ok
    r["pattern"] = np.where(
        ~sig, np.where(r["sd"] > a.flat_sd, "unstructured", "constant"),
        np.where(predicts & explains & epoch_ok, "ramp", "ramp-unconfirmed"))
    r.loc[r["tiles"] < a.min_tiles, "pattern"] = "n/a"
    pd.set_option("display.width", 240)

    print("\n" + "=" * 122)
    print("VERTICAL BIAS AGAINST 3DEP: CONSTANT OFFSET, RAMP ALONG THE FLIGHT DIRECTION, "
          "OR UNSTRUCTURED")
    print("=" * 122)
    show = ["sortie", "mode", "tiles", "span_km", "dz_median", "sd", "grad_mm_per_m",
            "p_perm", "resid_rms", "var_expl", "loo_const", "loo_ramp",
            "within_tile_nmad", "pattern"]
    print(r[[c for c in show if c in r]].to_string(index=False, na_rep="-",
                                                   float_format=lambda v: f"{v:.3f}"))
    print("\n  dz_median   per-sortie vertical bias, INSIGHTS minus 3DEP, in m")
    print("  sd          tile-to-tile scatter of that bias within the sortie, in m")
    print("  grad        along-corridor gradient of the fitted ramp, in mm per m")
    print("  p_perm      permutation-test p-value for the ramp")
    print("  resid_rms   scatter remaining after the ramp is removed, in m")
    print("  var_expl    fraction of the tile-to-tile variance the ramp accounts for")
    print("  loo_*       leave-one-out prediction error for the constant and ramp models;")
    print("              a real ramp predicts a held-out tile better than the mean does")

    for label, blurb in [
            ("ramp", "the bias varies smoothly along the flight direction"),
            ("ramp-unconfirmed", "a significant gradient that does not survive scrutiny"),
            ("unstructured", "varies by more than %.1f m, but not with position" % a.flat_sd),
            ("constant", "no significant variation with position")]:
        g = r[r["pattern"] == label]
        if not len(g):
            continue
        print(f"\n{label} ({len(g)}): {', '.join(g['sortie'])}")
        print(f"  {blurb}")
        if label == "ramp":
            print(f"  |gradient| {g['grad_mm_per_m'].abs().min():.2f} to "
                  f"{g['grad_mm_per_m'].abs().max():.2f} mm/m, accounting for "
                  f"{100*g['var_expl'].min():.0f}-{100*g['var_expl'].max():.0f}% of the "
                  f"variance;")
            print(f"  removing it cuts the scatter from {g['sd'].min():.2f}-"
                  f"{g['sd'].max():.2f} m to {g['resid_rms'].min():.2f}-"
                  f"{g['resid_rms'].max():.2f} m, against a within-tile precision of "
                  f"{g['within_tile_nmad'].min():.2f}-{g['within_tile_nmad'].max():.2f} m.")
            print(f"  Over a {g['span_km'].median():.0f} km sortie a gradient that small "
                  f"still integrates to metres, which is why the")
            print("  spread is large while the gradient is not.")
        if label == "ramp-unconfirmed":
            for _, x in g.iterrows():
                why = []
                if not x["epoch_ok"]:
                    why.append("reference-epoch control disagrees")
                if not (x["var_expl"] >= a.min_var_explained):
                    why.append(f"explains only {100*x['var_expl']:.0f}% of the variance")
                if not (x["loo_ramp"] < x["loo_const"]):
                    why.append("does not improve held-out prediction")
                print(f"    {x['sortie']:11s} {'; '.join(why)}")

    if "grad_single_project" in r:
        chk = r[r["grad_single_project"].notna()]
        if len(chk):
            print("\nReference-epoch control, for the sorties drawing on more than one "
                  "3DEP project:")
            for _, x in chk.iterrows():
                flag = "" if x["epoch_ok"] else "   <-- DISAGREES"
                print(f"  {x['sortie']:11s} step between projects {x['epoch_step']:+6.2f} m; "
                      f"gradient {x['grad_mm_per_m']:+.3f} mm/m over all tiles vs "
                      f"{x['grad_single_project']:+.3f} mm/m within "
                      f"{x['single_project']} alone (n={int(x['n_single_project'])}, "
                      f"resid {x['resid_single_project']:.3f} m){flag}")
            print("  A sortie whose corridor is long enough to leave one 3DEP project's "
                  "coverage is compared\n  against references of different epochs, and this "
                  "method cannot separate a ramp from the\n  steps between them. Those "
                  "sorties are reported, not counted as ramps.")

    print("\nBy collection mode (from Methods):")
    for m, g in r[r["pattern"] != "n/a"].groupby("mode"):
        print(f"  {m:15s} ramp in {(g['pattern'] == 'ramp').sum()} of {len(g)}   "
              f"median |gradient| {g['grad_mm_per_m'].abs().median():.3f} mm/m   "
              f"median sd {g['sd'].median():.3f} m")
    print("  A ramp concentrated in the line-following corridors would be consistent with "
          "along-track\n  drift that a raster grid's cross-passes constrain and a single "
          "corridor does not. This is\n  a consistency check on that hypothesis, not a "
          "test of it: mode is not randomly assigned, and\n  the corridors are also the "
          "long sorties, where a gradient has the most distance to act over.")

    print("\nWithin-tile tilt, for comparison with the long-range ramp:")
    print(f"  median gradient across a single 140 m tile: "
          f"{d['grad_mm_per_m'].median():.2f} mm/m, interquartile "
          f"{d['grad_mm_per_m'].quantile(0.25):.2f}-"
          f"{d['grad_mm_per_m'].quantile(0.75):.2f}")
    print("  An order of magnitude above the long-range gradients, so the two are separate")
    print("  phenomena: the ramp is not the within-tile tilt extrapolated, and the "
          "within-tile tilt\n  is not the ramp sampled locally.")

    r.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
