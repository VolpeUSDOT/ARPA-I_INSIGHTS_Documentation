#!/usr/bin/env python3
"""Release-wide footprint area recomputation.

Sums the STAC footprint polygon areas of all 82,429 tiles and cross-checks that
sum against several other area conventions, so that the manuscript's
"1610.50 km^2 / 87.00 points/m^2" pair can be traced to a single, stated
convention.

The authoritative number is the sum of the *footprint polygons* reprojected to an
equal-area CRS (EPSG:5070, Albers Equal Area CONUS) and integrated with the
shoelace formula.  Every other row printed here is a foil: it is reported so that
a reviewer can see how far each alternative convention drifts, and why it is not
the released area.

Run with --help for options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from shapely import wkb

# US survey foot.  Both projected CRSs used by this release (EPSG:6430 Colorado
# North and EPSG:6626 Utah North) are in US survey feet, NOT international
# feet.  The two differ by 2 ppm, which is 3 km^2 over a 1.6 Mkm^2 release -- not
# enough to matter here, but the wrong constant silently biases every length.
M_PER_FTUS = 0.3048006096012192

# Foils only (see the table this script prints).  These are NOT the footprint
# geometry: the actual STAC footprint polygons measure ~139.96 m on a side.
NOMINAL_FOOTPRINT_SIDE_M = 140.836
GRID_PITCH_M = 139.957
MEDIAN_PROJ_BBOX_WIDTH_FTUS = 462.2555

INDEX_RELPATH = Path("data/lidar/v1/stac/index/items.parquet")


def default_index_path() -> Path:
    """Resolve the index relative to the repo root, never to an absolute
    scratch path, so the scripts are portable across checkouts."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / INDEX_RELPATH
        if candidate.exists():
            return candidate
    # Fall back to the layout implied by validation/coverage/<this file>.
    return here.parents[2] / INDEX_RELPATH


def shoelace(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorised polygon area for an (n, k) array of ring vertices."""
    return 0.5 * np.abs(
        np.sum(xs * np.roll(ys, -1, axis=1) - np.roll(xs, -1, axis=1) * ys, axis=1)
    )


def load_corners(index_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (frame, corners) where corners is (n, 4, 2) lon/lat.

    TRAP: `geometry` is WKB holding a single-part MultiPolygon whose exterior
    ring has SIX coordinates -- the four corners, the closing vertex, and a
    duplicate of that closing vertex -- and the coordinates are XYZ
    (lon, lat, elevation), not XY.  Taking all six vertices, or forgetting the
    third ordinate, silently corrupts the shoelace sum.  Take the first four
    and drop Z.
    """
    df = pd.read_parquet(
        index_path,
        columns=["id", "sortie", "proj:epsg", "proj:bbox", "pc:count", "geometry"],
    )
    geoms = wkb.loads(df["geometry"].values)
    rings = []
    for g in geoms:
        assert g.geom_type == "MultiPolygon" and len(g.geoms) == 1
        coords = np.asarray(g.geoms[0].exterior.coords)
        assert coords.shape == (6, 3), f"unexpected ring shape {coords.shape}"
        rings.append(coords[:4, :2])
    return df.drop(columns=["geometry"]), np.asarray(rings)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Recompute the release-wide footprint area.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--index", type=Path, default=default_index_path(),
                    help="path to items.parquet")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON file to write the computed areas to")
    ap.add_argument("--skip-geodesic", action="store_true",
                    help="skip the per-tile geodesic cross-check (the slow part, ~30 s)")
    args = ap.parse_args(argv)

    if not args.index.exists():
        ap.error(f"index not found: {args.index}")

    df, corners = load_corners(args.index)
    n_tiles = len(df)
    print(f"index: {args.index}")
    print(f"tiles: {n_tiles}")

    # --- authoritative: footprint polygons in an equal-area CRS -------------
    tr5070 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x5070, y5070 = tr5070.transform(corners[:, :, 0], corners[:, :, 1])
    area_5070 = shoelace(np.asarray(x5070), np.asarray(y5070))
    km2_5070 = area_5070.sum() / 1e6

    # --- same polygons, native State Plane (sanity: projection choice is not
    #     doing the work) ---------------------------------------------------
    native_m2 = 0.0
    for epsg in sorted(df["proj:epsg"].unique()):
        mask = (df["proj:epsg"] == epsg).to_numpy()
        tr = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        xs, ys = tr.transform(corners[mask, :, 0], corners[mask, :, 1])
        # State Plane here is in US survey FEET; convert before taking area.
        native_m2 += shoelace(np.asarray(xs) * M_PER_FTUS,
                              np.asarray(ys) * M_PER_FTUS).sum()
    km2_native = native_m2 / 1e6

    # --- geodesic cross-check on the ellipsoid ------------------------------
    km2_geod = None
    if not args.skip_geodesic:
        geod = pyproj.Geod(ellps="GRS80")
        total = 0.0
        for i in range(n_tiles):
            total += abs(geod.polygon_area_perimeter(corners[i, :, 0],
                                                    corners[i, :, 1])[0])
        km2_geod = total / 1e6

    # --- foil 1: sum of proj:bbox rectangles -------------------------------
    # TRAP: proj:bbox is the axis-aligned bounding box of the RETURNS, clipped to
    # the data, and several sortie grids are rotated 10-29 deg off the State
    # Plane axes.  For those sorties the bbox circumscribes the tile, so this sum
    # is inflated -- it is not a footprint area under any convention.
    pbbox = np.array([json.loads(s) for s in df["proj:bbox"]])
    bbox_m2 = ((pbbox[:, 2] - pbbox[:, 0]) * (pbbox[:, 3] - pbbox[:, 1])
               * M_PER_FTUS * M_PER_FTUS)
    km2_bbox = bbox_m2.sum() / 1e6

    # --- foils 2-4: count x a single nominal cell size ----------------------
    # TRAP: these all overshoot, because edge-tile footprints are CLIPPED.  The
    # 462.2555 ft figure in particular is the median proj:bbox WIDTH, not a
    # footprint side; using it as "one full cell" overstates the area.
    km2_full = n_tiles * NOMINAL_FOOTPRINT_SIDE_M ** 2 / 1e6
    km2_pitch = n_tiles * GRID_PITCH_M ** 2 / 1e6
    side_from_bbox_m = MEDIAN_PROJ_BBOX_WIDTH_FTUS * M_PER_FTUS
    km2_bboxside = n_tiles * side_from_bbox_m ** 2 / 1e6

    total_points = int(df["pc:count"].sum())
    density = total_points / area_5070.sum()
    implied_mean_side = float(np.sqrt(area_5070.mean()))

    print()
    print("area by convention")
    print("-" * 62)
    rows = [
        ("footprint polygons, equal-area (EPSG:5070)", km2_5070, "AUTHORITATIVE"),
        ("footprint polygons, native State Plane", km2_native, ""),
    ]
    if km2_geod is not None:
        rows.append(("footprint polygons, geodesic (GRS80)", km2_geod, ""))
    rows += [
        ("sum of proj:bbox rectangles", km2_bbox, "inflated by rotated grids"),
        (f"{n_tiles} x full footprint ({NOMINAL_FOOTPRINT_SIDE_M} m)", km2_full,
         "edge tiles are clipped"),
        (f"{n_tiles} x grid pitch ({GRID_PITCH_M} m)", km2_pitch,
         "edge tiles are clipped"),
        (f"{n_tiles} x ({MEDIAN_PROJ_BBOX_WIDTH_FTUS} ft)^2", km2_bboxside,
         "bbox width is not a footprint side"),
    ]
    for label, km2, note in rows:
        print(f"  {label:<48s} {km2:10.2f} km^2  {note}")

    if km2_geod is not None:
        print()
        print(f"geodesic / equal-area ratio: {km2_geod / km2_5070:.6f}")

    print()
    print(f"total points:            {total_points:,}")
    print(f"footprint area:          {km2_5070:.2f} km^2")
    print(f"release point density:   {density:.2f} points/m^2")
    print()
    print(f"implied mean footprint side: {implied_mean_side:.3f} m "
          f"(vs {NOMINAL_FOOTPRINT_SIDE_M} m nominal -> "
          f"{100 * (1 - implied_mean_side ** 2 / NOMINAL_FOOTPRINT_SIDE_M ** 2):.1f}% "
          f"area shortfall, {km2_full - km2_5070:.1f} km^2)")
    print()
    print("CONCLUSION: only the polygon sum is the released footprint area. The "
          "count-times-nominal-cell family (1614-1636 km^2) overshoots because "
          "edge-tile footprints are clipped, and the proj:bbox sum overshoots "
          "further because rotated grids make the axis-aligned box circumscribe "
          "the tile.")

    if args.out is not None:
        payload = {
            "index": str(args.index),
            "n_tiles": n_tiles,
            "km2_footprint_equal_area_5070": km2_5070,
            "km2_footprint_native_stateplane": km2_native,
            "km2_footprint_geodesic_grs80": km2_geod,
            "km2_sum_proj_bbox": km2_bbox,
            "km2_count_x_full_footprint": km2_full,
            "km2_count_x_grid_pitch": km2_pitch,
            "km2_count_x_median_bbox_width": km2_bboxside,
            "total_points": total_points,
            "points_per_m2": density,
            "implied_mean_footprint_side_m": implied_mean_side,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
