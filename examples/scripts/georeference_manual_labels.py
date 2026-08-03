#!/usr/bin/env python3
"""
Restore projected coordinates to INSIGHTS-Manual-Semantic-Labels tiles.

Why this is needed:
  The released manual-semantic label tiles are distributed in a normalized coordinate
  frame, ready for machine-learning use: each tile is centered on its own footprint
  and uniformly scaled so its longest axis spans 100 units. They therefore carry no
  coordinate reference system, and cannot be overlaid on the LiDAR, on GIS layers, or
  on each other as delivered.

  The normalization is a similarity transform, so it inverts exactly. This script
  applies the inverse and writes a tile that is georeferenced in the sortie's native
  projected CRS (EPSG:6430 for Colorado, EPSG:6626 for Utah), with the CRS recorded
  in the output header.

  Per-tile transform parameters come from the label product's GeoParquet index
  (the `norm:*` columns), so nothing has to be re-derived here:

      X = x_norm * norm:scale + norm:offset_x
      Y = y_norm * norm:scale + norm:offset_y
      Z = z_norm * norm:scale + norm:offset_z

What is and is not recovered:
  Classification labels and point geometry are recovered exactly, to within the
  0.01-unit quantization of the normalized tiles (about 0.04 US survey feet).

  Intensity, GPS time, return numbers, scan angle, and point source ID are *not*
  recovered: the annotation pipeline zeroed them. Read those from the unlabeled
  source tile instead, at the `source_lidar_https_path` column of the same index row.
  Because the label tile and the source tile hold the same point set, labels can be
  transferred onto the source tile by nearest-neighbour lookup after georeferencing.

Output format:
  laspy writes LAS and LAZ. COPC output additionally requires PDAL on PATH; pass
  --copc to run `pdal translate` as a final step.

Requirements:
  laspy, lazrs, pyproj, geopandas, pyarrow. PDAL only for --copc.

Example:
  python examples/scripts/georeference_manual_labels.py \
      --in data/insights/manual_semantic_tiles \
      --out data/insights/manual_semantic_georeferenced \
      --copc
"""

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import geopandas as gpd
import laspy
import numpy as np
import pyproj

DEFAULT_INDEX = (
    "https://arpa-i-insights.s3.amazonaws.com/labels/manual-semantic/v1/index/"
    "manual-semantic-labels-index.geoparquet"
)

# Coordinate scale for the output, in the CRS's own linear unit (US survey feet).
# Matches the scale used by the canonical INSIGHTS-LiDAR tiles.
OUTPUT_SCALE = 0.01

TILE_SUFFIX = ".copc.laz"


def read_uri(uri: str) -> bytes:
    """Read a local path or an http(s) URL into memory."""
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri) as response:
            return response.read()
    return Path(uri).read_bytes()


def load_index(uri: str) -> gpd.GeoDataFrame:
    """Load the manual-semantic label index, keyed by tile file-name stem."""
    index = gpd.read_parquet(io.BytesIO(read_uri(uri)))
    required = {"file_name_stem", "norm:scale", "norm:offset_x", "norm:offset_y",
                "norm:offset_z", "proj:epsg"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Index at {uri} is missing required columns: {sorted(missing)}")
    return index.set_index("file_name_stem")


def stem_of(path: Path) -> str:
    """Strip the compound .copc.laz suffix, which Path.stem only halves."""
    name = path.name
    if name.endswith(TILE_SUFFIX):
        return name[: -len(TILE_SUFFIX)]
    return path.stem


def georeference(
    in_path: Path,
    out_path: Path,
    scale: float,
    offset_x: float,
    offset_y: float,
    offset_z: float,
    epsg: int,
) -> int:
    """
    Write a georeferenced copy of one normalized label tile.

    Returns:
        The number of points written.
    """
    source = laspy.read(in_path)

    x = np.asarray(source.x) * scale + offset_x
    y = np.asarray(source.y) * scale + offset_y
    z = np.asarray(source.z) * scale + offset_z

    header = laspy.LasHeader(version=source.header.version, point_format=source.header.point_format)
    header.scales = np.array([OUTPUT_SCALE, OUTPUT_SCALE, OUTPUT_SCALE])
    # Integer offsets near the data keep the scaled integer coordinates well within range.
    header.offsets = np.floor([x.min(), y.min(), z.min()])
    header.add_crs(pyproj.CRS.from_epsg(epsg))

    output = laspy.LasData(header)
    # Copy every dimension the source carries, then overwrite the coordinates. Most
    # ancillary dimensions are zero in this product, but copying keeps the script
    # correct if a future release restores them.
    for dimension in source.point_format.dimension_names:
        if dimension in ("X", "Y", "Z"):
            continue
        setattr(output, dimension, getattr(source, dimension))
    output.x, output.y, output.z = x, y, z

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.write(out_path)
    return len(x)


def to_copc(laz_path: Path, copc_path: Path) -> None:
    """Rewrite a LAS/LAZ file as COPC using the PDAL CLI."""
    if shutil.which("pdal") is None:
        raise SystemExit(
            "--copc requires the `pdal` CLI on PATH. Install PDAL, or drop --copc to "
            "write plain LAZ."
        )
    command = [
        "pdal", "translate", str(laz_path), str(copc_path), "--writers.copc.forward=all",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"PDAL failed to write COPC (exit={error.returncode}).\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}\n"
        ) from error


def iter_inputs(in_path: Path) -> list[Path]:
    """Resolve the input argument to a sorted list of LAS/LAZ files."""
    in_path = in_path.expanduser().resolve()
    if in_path.is_file():
        return [in_path]
    if not in_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {in_path}")
    suffixes = {".las", ".laz"}
    return sorted(p for p in in_path.rglob("*") if p.suffix.lower() in suffixes)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore projected coordinates to normalized manual-semantic label tiles.",
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="Input label tile, or a directory searched recursively for .las/.laz files.",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help="Manual-semantic label GeoParquet index; a local path or an https URL.",
    )
    parser.add_argument(
        "--copc",
        action="store_true",
        help="Also write COPC via the `pdal` CLI (requires PDAL on PATH).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    in_files = iter_inputs(args.in_path)
    if not in_files:
        raise SystemExit(f"No .las/.laz files found under: {args.in_path}")

    index = load_index(args.index)
    out_dir: Path = args.out_dir.expanduser().resolve()

    failures = 0
    for in_file in in_files:
        stem = stem_of(in_file)
        if stem not in index.index:
            print(f"SKIP {stem}: not present in the label index", file=sys.stderr)
            failures += 1
            continue

        row = index.loc[stem]
        epsg = int(row["proj:epsg"])
        laz_path = out_dir / f"{stem}.laz"
        copc_path = out_dir / f"{stem}{TILE_SUFFIX}"
        final_path = copc_path if args.copc else laz_path

        if final_path.exists() and not args.overwrite:
            print(f"skip {stem}: {final_path.name} exists (use --overwrite)")
            continue

        n_points = georeference(
            in_path=in_file,
            out_path=laz_path,
            scale=float(row["norm:scale"]),
            offset_x=float(row["norm:offset_x"]),
            offset_y=float(row["norm:offset_y"]),
            offset_z=float(row["norm:offset_z"]),
            epsg=epsg,
        )

        if args.copc:
            to_copc(laz_path, copc_path)
            laz_path.unlink()

        print(f"wrote {final_path.name}  {n_points:,} points  EPSG:{epsg}")

    if failures:
        raise SystemExit(f"{failures} of {len(in_files)} file(s) were skipped.")


if __name__ == "__main__":
    main()
