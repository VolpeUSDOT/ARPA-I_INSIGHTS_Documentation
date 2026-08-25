"""Horizontal agreement between overlapping sorties, from the point clouds alone.

`characterize_offsets.py` measures how two overlapping sorties disagree *vertically*, and it
is deliberately blind to horizontal disagreement: it restricts itself to flat ground, where
a horizontal misregistration produces almost no vertical difference. That restriction is why
its vertical numbers are trustworthy, and it is also why the Usage Notes have to concede that
horizontal agreement between sorties is unmeasured. This script measures it.

`coregister_3dep.py` answers the neighbouring question -- how far each sortie sits from an
external reference -- and needs terrain that the flat urban overlaps often lack. This one
compares the two sorties against *each other*, so it is unaffected by any error in an
external product and by any real change since an external product's epoch. What it cannot do
is say which of the two sorties is wrong.

The method
----------
The same relation as in `coregister_3dep.py`. Over the overlap, with dh the difference of the
two sorties' ground surfaces,

    dh = h_a - h_b = dz + dx * dH/dx + dy * dH/dy + beta * |grad H|

and `(dx, dy)` is the translation that must be **applied to sortie A** to bring it onto
sortie B, A being the alphabetically first of the pair so the sign is well defined.

Two things differ from the external comparison, both in this script's favour:

* **The ground-proxy bias cancels.** Both surfaces are estimated by the same low percentile
  on the same grid, so the downhill bias that term corrects for is common to both and
  differences away. The term is kept anyway, and a fitted beta near zero is a check that the
  cancellation really happens rather than an assumption that it does.
* **No epoch gap.** Both sorties were flown in June 2025, so real change contributes nothing.

One thing is worse: the gradient has to come from somewhere, and taking it from either sortie
correlates the regressor with the noise in the response, which biases the fit. Taking it from
the average of the two would cancel that only if both carry the same noise, which across this
release they demonstrably do not. So the gradient comes from the 3DEP 1 m DEM, which is
independent of both. The DEM contributes its own noise to the regressor -- attenuating the
estimate slightly -- but contributes nothing to dh, so it cannot bias the sign or invent a
shift.

Conditioning and uncertainty are handled exactly as in `coregister_3dep.py`: a pooled fit per
sortie combination carrying one vertical offset per tile pair, since the vertical offset
varies from place to place within a combination; a formal standard error inflated by the
scatter between pairs, because cells within an overlap are strongly correlated and the raw
per-cell error is meaningless; and an explicit conditioning report, because a shift is
identified by variation in slope and aspect rather than by slope alone.

Excluded, following `characterize_offsets.py`: the western third of `I80P2` (gx < 27), whose
dense noise would corrupt a ground estimate.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import grey_dilation, grey_erosion, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import characterize_offsets as CO
import compare_3dep as C3
import coregister_3dep as CG

M_PER_FTUS = CO.M_PER_FTUS
CELL_M = 2.0
MIN_PTS_PER_CELL = 20
BASE_SPREAD_M = 0.5
OPEN_CELLS = 21
BARE_TOL_M = 0.5
SMOOTH_CELLS = 3
MIN_FIT_CELLS = 300      # per pair; overlaps are smaller than whole tiles


def surface(x, y, z, x0, y0, nx, ny, grad_mag):
    """Ground proxy and usability mask for one sortie over the overlap grid."""
    cell = CELL_M / M_PER_FTUS
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[m], iy[m], z[m]
    if not len(z):
        return None, None
    f = ix * ny + iy
    o = np.argsort(f, kind="stable")
    fs, zs = f[o], z[o]
    b = np.searchsorted(fs, np.arange(nx * ny + 1))
    cnt = np.diff(b)
    g = np.full(nx * ny, np.nan)
    sp = np.full(nx * ny, np.nan)
    for c in np.nonzero(cnt >= MIN_PTS_PER_CELL)[0]:
        seg = zs[b[c]:b[c + 1]]
        p10, p50 = np.percentile(seg, [10, 50])
        g[c], sp[c] = p10, p50 - p10
    g = g.reshape(nx, ny)
    sp = sp.reshape(nx, ny) * M_PER_FTUS
    have = np.isfinite(g)
    # slope-aware spread allowance: sloped ground must not be rejected here, since slope
    # is what carries the horizontal signal
    clean = have & (sp < BASE_SPREAD_M + grad_mag * CELL_M)
    filled = np.where(have, g, np.nanmax(g) if have.any() else 0.0)
    opened = grey_dilation(grey_erosion(filled, size=(OPEN_CELLS, OPEN_CELLS)),
                           size=(OPEN_CELLS, OPEN_CELLS))
    gy, gx = np.gradient(opened * M_PER_FTUS, CELL_M)
    drop = np.hypot(gx, gy) * (OPEN_CELLS * CELL_M / 2.0)
    bare = (g - opened) * M_PER_FTUS <= (BARE_TOL_M + drop)
    return g, clean & bare


def measure(r, ref, use_slope_term):
    cell = CELL_M / M_PER_FTUS
    nx = max(int((r["x1"] - r["x0"]) / cell), 1)
    ny = max(int((r["y1"] - r["y0"]) / cell), 1)
    base = dict(combo=r["combo"], a=r["a"], b=r["b"], sa=r["sa"], sb=r["sb"])
    if nx < 20 or ny < 20:
        return dict(base, ok=False, note="overlap too small")

    cx = r["x0"] + (np.arange(nx) + 0.5) * cell
    cy = r["y0"] + (np.arange(ny) + 0.5) * cell
    GX, GY = np.meshgrid(cx, cy, indexing="ij")
    src = f"EPSG:{int(r['epsg'])}"
    lon, lat = Transformer.from_crs(src, "EPSG:4326", always_xy=True).transform(
        float(GX.mean()), float(GY.mean()))

    # terrain gradient from 3DEP, independent of both sorties
    grad = None
    tried = []
    for demkey, proj, year in ref.candidates(lon, lat):
        def to_dem(wkt, _gx=GX, _gy=GY):
            return Transformer.from_crs(src, wkt, always_xy=True).transform(
                _gx.ravel(), _gy.ravel())
        got = CG.sample_dem_bilinear(demkey, src, to_dem)
        if isinstance(got, str):
            tried.append(f"{proj}:{got}")
            continue
        rz, rvalid, epsg = got
        R = rz.reshape(nx, ny); RV = rvalid.reshape(nx, ny)
        Rf = np.where(RV, R, 0.0)
        w = uniform_filter(RV.astype(float), SMOOTH_CELLS, mode="nearest")
        Rs = np.where(w > 0, uniform_filter(Rf, SMOOTH_CELLS, mode="nearest") /
                      np.maximum(w, 1e-9), np.nan)
        Hx, Hy = np.gradient(Rs, CELL_M, CELL_M)
        grad = (Hx, Hy, np.hypot(Hx, Hy), proj, year)
        break
    if grad is None:
        return dict(base, ok=False, note="no usable 3DEP gradient",
                    skipped=";".join(tried))
    Hx, Hy, gmag, proj, year = grad

    xa, ya, za = CO.fetch(r["pa"])
    ga, ua = surface(xa, ya, za, r["x0"], r["y0"], nx, ny, np.nan_to_num(gmag))
    del xa, ya, za
    xb, yb, zb = CO.fetch(r["pb"])
    gb, ub = surface(xb, yb, zb, r["x0"], r["y0"], nx, ny, np.nan_to_num(gmag))
    del xb, yb, zb
    if ga is None or gb is None:
        return dict(base, ok=False, note="no points in overlap")

    sel = ua & ub & np.isfinite(Hx) & np.isfinite(Hy)
    if int(sel.sum()) < MIN_FIT_CELLS:
        return dict(base, ok=False, note=f"too few co-observed cells ({int(sel.sum())})")

    # A is the alphabetically first sortie, so the sign of the answer is well defined
    swap = r["sa"] != r["lo"]
    dh = (gb[sel] - ga[sel] if swap else ga[sel] - gb[sel]) * M_PER_FTUS
    hx, hy, gm = Hx[sel], Hy[sel], gmag[sel]
    cols = [np.ones(dh.size), hx, hy] + ([gm] if use_slope_term else [])
    A = np.column_stack(cols)
    fit = CG.robust_fit(A, dh)
    if fit is None:
        return dict(base, ok=False, note="fit failed")
    coef, cov, nkeep, sig = fit
    se = np.sqrt(np.diag(cov))
    return dict(base, ok=True, lo=r["lo"], hi=r["hi"], dem_project=proj, dem_year=year,
                n=int(sel.sum()), n_fit=nkeep,
                dz_m=round(float(coef[0]), 4),
                dx_m=round(float(coef[1]), 4), dy_m=round(float(coef[2]), 4),
                se_dx_m=round(float(se[1]), 4), se_dy_m=round(float(se[2]), 4),
                beta_m=round(float(coef[3]), 4) if use_slope_term else None,
                resid_sigma_m=round(sig, 4),
                grad_p50=round(float(np.median(gm)), 4),
                cross=CG.cross_products(dh, cols[1:], A, coef),
                x0=round(float(r["x0"]), 1), y0=round(float(r["y0"]), 1))


def report(path, use_slope_term):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows).drop_duplicates(subset=["a", "b"], keep="last")
    good = d[d.ok == True] if "ok" in d else d
    pd.set_option("display.width", 220)
    print(f"\nusable pairs: {len(good)} of {len(d)} attempted")
    if "note" in d and d.ok.eq(False).any():
        print("\nrejections:")
        print(d[d.ok == False].groupby("note").size().to_string())
    if not len(good):
        return pd.DataFrame()

    recs = []
    for combo, g in good.groupby("combo"):
        p = CG.pooled(g.to_dict("records"), use_slope_term)
        if p is None:
            continue
        infl = {}
        for ax in ("dx", "dy"):
            est = g[f"{ax}_m"].to_numpy(float)
            sef = g[f"se_{ax}_m"].to_numpy(float)
            m = np.isfinite(est) & np.isfinite(sef) & (sef > 0)
            infl[ax] = (max(np.sqrt(float((((est[m] - p[ax]) / sef[m]) ** 2).sum())
                                    / (m.sum() - 1)), 1.0) if m.sum() >= 2 else np.nan)
        se_dx, se_dy = p["se_dx"] * infl["dx"], p["se_dy"] * infl["dy"]
        sd_x = float(1.4826 * np.median(np.abs(g["dx_m"] - np.median(g["dx_m"]))))
        sd_y = float(1.4826 * np.median(np.abs(g["dy_m"] - np.median(g["dy_m"]))))
        se_tile = max(sd_x, sd_y) / np.sqrt(max(p["tiles"], 1))
        mag = float(np.hypot(p["dx"], p["dy"]))
        se_mag = max(float(np.hypot(p["dx"] * se_dx, p["dy"] * se_dy) / max(mag, 1e-9)),
                     se_tile)
        recs.append(dict(combo=combo, pairs=p["tiles"], cells=p["cells"],
                         dx_m=p["dx"], dy_m=p["dy"], se_dx=se_dx, se_dy=se_dy,
                         sd_pair_x=sd_x, sd_pair_y=sd_y,
                         mag_m=mag, se_mag=se_mag, snr=mag / max(se_mag, 1e-9),
                         beta_m=p["beta"], sigma_m=p["sigma"],
                         gx_gy_corr=p["gx_gy_corr"],
                         grad_p50=float(g["grad_p50"].median())))
    r = pd.DataFrame(recs).sort_values("combo").reset_index(drop=True)

    print("\n" + "=" * 118)
    print("HORIZONTAL SHIFT BETWEEN OVERLAPPING SORTIES")
    print("pooled per combination, one vertical offset per tile pair, shared (dx, dy)")
    print("=" * 118)
    cols = ["combo", "pairs", "cells", "dx_m", "se_dx", "dy_m", "se_dy", "mag_m",
            "se_mag", "snr", "sd_pair_x", "grad_p50", "sigma_m", "gx_gy_corr", "beta_m"]
    print(r[cols].to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.3f}"))
    print("\n  dx, dy      translation to apply to the FIRST-NAMED sortie to bring it onto")
    print("              the second, in m")
    print("  se_*        formal error inflated by the scatter between pairs; the raw")
    print("              per-cell error is meaningless, cells in an overlap being correlated")
    print("  sd_pair_x   robust scatter of the per-pair estimates, m")
    print("  snr         magnitude over its own standard error; below about 3 the terrain")
    print("              did not constrain the estimate")
    print("  beta_m      the ground-proxy bias term. It should be near zero here, because")
    print("              both surfaces use the same estimator and the bias differences away")

    strong = r[r.snr >= 3]
    print(f"\n{len(strong)} of {len(r)} combinations resolve a horizontal shift "
          f"(snr >= 3){':' if len(strong) else '.'}")
    if len(strong):
        print(f"  {', '.join(strong['combo'])}")
        print(f"  |shift| {strong.mag_m.min():.2f} to {strong.mag_m.max():.2f} m, "
              f"median {strong.mag_m.median():.2f} m")
        print("  These are DIFFERENCES between two sorties, so a shift of this size means "
              "the two\n  disagree horizontally by that much; it does not say which one is "
              "displaced.")
    weak = r[r.snr < 3]
    if len(weak):
        print(f"\n{len(weak)} combinations do not resolve one: {', '.join(weak['combo'])}")
        print("  Read that as insufficient leverage, not as agreement.")
    if use_slope_term and r["beta_m"].notna().any():
        print(f"\nGround-proxy bias term: median beta {r['beta_m'].median():+.3f} m per unit "
              f"gradient,\n  against {-0.4 * CELL_M:+.2f} m for a single surface in "
              f"coregister_3dep.py. Near zero confirms\n  that the bias cancels between two "
              f"surfaces estimated the same way.")
    return r


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(repo, "data", "lidar", "v1", "stac",
                                                    "index", "items.parquet"))
    ap.add_argument("--per-combo", type=int, default=8,
                    help="tile pairs per sortie combination, spread along the overlap")
    ap.add_argument("--min-overlap", type=float, default=5000.0,
                    help="minimum bbox overlap area, m^2")
    ap.add_argument("--min-relief", type=float, default=15.0,
                    help="skip pairs whose weaker tile has less index relief than this, m, "
                         "since flat overlaps cannot inform a horizontal shift")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--combo", action="append")
    ap.add_argument("--no-slope-term", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(here, "out", "coregister_pairs.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    use_slope = not a.no_slope_term

    if not a.report_only:
        d = CO.load_index(a.index)
        n0 = len(d)
        d = d[~((d["sortie"] == "I80P2") & (d["gx"] < 27))]
        print(f"excluded {n0 - len(d)} tiles: western third of I80P2 (dense noise)",
              file=sys.stderr)
        p = CO.candidate_pairs(d, a.min_overlap)
        # prefer overlaps with terrain, for the same reason coregister_3dep.py does
        rel = pd.read_parquet(a.index, columns=["id", "elevation:min", "elevation:max"])
        rel["relief_m"] = (rel["elevation:max"] - rel["elevation:min"]) * M_PER_FTUS
        rm = rel.set_index("id")["relief_m"]
        p["relief_m"] = np.minimum(p["a"].map(rm), p["b"].map(rm))
        n1 = len(p)
        p = p[p["relief_m"] >= a.min_relief]
        print(f"{len(p)} of {n1} candidate pairs have >= {a.min_relief} m relief",
              file=sys.stderr)
        p = p.merge(d[["id", "epsg"]].rename(columns={"id": "a"}), on="a", how="left")
        samp = CO.spread_sample(p, a.per_combo, 0)
        if a.combo:
            samp = samp[samp["combo"].isin(a.combo)]
        print("indexing 3DEP 1 m projects", file=sys.stderr)
        ref = C3.Reference(C3.build_index(
            os.path.join(os.path.dirname(a.out), "dem_index.json")))
        done = set()
        if os.path.exists(a.out):
            done = {(json.loads(l).get("a"), json.loads(l).get("b")) for l in open(a.out)}
        todo = samp[~samp.apply(lambda x: (x["a"], x["b"]) in done, axis=1)] \
            if done else samp
        print(f"measuring {len(todo)} tile pairs with {a.workers} workers", file=sys.stderr)
        errs = []
        with open(a.out, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(measure, r, ref, use_slope): (r["a"], r["b"])
                    for _, r in todo.iterrows()}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    fh.write(json.dumps(f.result()) + "\n"); fh.flush()
                except Exception as e:
                    errs.append((futs[f], repr(e)[:200]))
                if i % 20 == 0:
                    print(f"  {i}/{len(futs)}  ({len(errs)} errors)", file=sys.stderr,
                          flush=True)
        for k, msg in errs:
            print(f"  err {k} {msg}")

    r = report(a.out, use_slope)
    if len(r):
        r.to_csv(a.out.replace(".jsonl", "_by_combo.csv"), index=False)
        print(f"\nwrote {a.out} and {a.out.replace('.jsonl', '_by_combo.csv')}")


if __name__ == "__main__":
    main()
