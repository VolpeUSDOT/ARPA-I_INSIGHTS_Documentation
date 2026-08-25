"""Pick tile pairs to inspect visually for each sortie pair, and emit COPC links.

For every sortie pair with measurements, selects two example tile pairs:

* **typical** -- the measured pair whose offset is closest to that sortie pair's median,
  so what you see is representative of the sortie pair as a whole.
* **extreme** -- the pair whose offset deviates most from that median. Where a sortie pair
  is internally consistent this will look much like the typical case, and the two rows
  being nearly identical is itself the finding. Where it is not, this is the pair worth
  understanding: either the offset genuinely varies with location, or that particular
  measurement is unreliable.

Ties are broken toward more comparable cells, so a selected pair is never one of the
thin-overlap measurements when a better one is available.

Output includes the overlap centroid in latitude and longitude, so a viewer can be pointed
straight at the ground being compared rather than at the whole tile.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer

M_PER_FTUS = 0.3048006096012192


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=os.path.join(here, "out", "offsets.jsonl"))
    ap.add_argument("--index", default=os.path.join(
        repo, "data", "lidar", "v1", "stac", "index", "items.parquet"))
    ap.add_argument("--out", default=None, help="write a CSV here as well")
    a = ap.parse_args()

    m = pd.DataFrame([json.loads(l) for l in open(a.pairs)])
    m = m[m.get("ok") == True].copy()

    idx = pd.read_parquet(a.index, columns=["id", "sortie", "https_path",
                                            "proj:bbox", "proj:epsg"])
    bb = np.array([json.loads(x) for x in idx["proj:bbox"]], dtype=float)
    idx["minx"], idx["miny"], idx["maxx"], idx["maxy"] = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    meta = idx.set_index("id")

    tr = {int(e): Transformer.from_crs(f"EPSG:{int(e)}", "EPSG:4326", always_xy=True)
          for e in idx["proj:epsg"].unique()}

    rows = []
    for combo, g in m.groupby("combo"):
        med = g["dz_m"].median()
        g = g.assign(dev=(g["dz_m"] - med).abs())
        typical = g.sort_values(["dev", "cells"], ascending=[True, False]).iloc[0]
        extreme = g.sort_values(["dev", "cells"], ascending=[False, False]).iloc[0]
        for kind, r in (("typical", typical), ("extreme", extreme)):
            ta, tb = meta.loc[r["a"]], meta.loc[r["b"]]
            x0, x1 = max(ta.minx, tb.minx), min(ta.maxx, tb.maxx)
            y0, y1 = max(ta.miny, tb.miny), min(ta.maxy, tb.maxy)
            lon, lat = tr[int(ta["proj:epsg"])].transform((x0 + x1) / 2, (y0 + y1) / 2)
            rows.append(dict(
                sortie_pair=combo, case=kind,
                dz_m=round(float(r["dz_m"]), 3),
                combo_median_dz_m=round(float(med), 3),
                deviation_m=round(float(r["dz_m"]) - float(med), 3),
                cells=int(r["cells"]),
                nmad_m=round(float(r["nmad_m"]), 3),
                tilt_mm_per_m=r["tilt_mm_per_m"],
                overlap_m2=int(r["overlap_m2"]),
                lat=round(lat, 6), lon=round(lon, 6),
                tile_a=r["a"], tile_b=r["b"],
                url_a=ta["https_path"], url_b=tb["https_path"]))
    out = pd.DataFrame(rows).sort_values(["sortie_pair", "case"], ascending=[True, False])

    for combo, g in out.groupby("sortie_pair", sort=True):
        print("=" * 78)
        print(f"{combo}    median offset {g['combo_median_dz_m'].iloc[0]:+.3f} m")
        print("=" * 78)
        for r in g.itertuples():
            print(f"  [{r.case}]  dz = {r.dz_m:+.3f} m   (deviation from median "
                  f"{r.deviation_m:+.3f} m)")
            print(f"      {r.cells} comparable cells, shape NMAD {r.nmad_m:.3f} m, "
                  f"tilt {r.tilt_mm_per_m} mm/m, overlap {r.overlap_m2} m2")
            print(f"      look near  {r.lat:.6f}, {r.lon:.6f}")
            print(f"      {r.tile_a}")
            print(f"        {r.url_a}")
            print(f"      {r.tile_b}")
            print(f"        {r.url_b}")
        print()

    if a.out:
        out.to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
