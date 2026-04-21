#!/usr/bin/env python3
"""
Classify ground in LAS/LAZ using PDAL, generate a ground DEM, and compute roughness + slope rasters.

Key behavior (units):
  - You specify --resolution and --roughness-window as strings with units, e.g. "10cm", "0.1m", "1ft", "1usft".
  - The script parses those into meters (canonical internal unit).
  - For each input LAS/LAZ file, the script attempts to infer the CRS linear unit (meters / feet / US survey feet)
    from PDAL metadata (WKT/PROJ4) and converts the requested meter values into the dataset's coordinate units
    before calling PDAL writers.gdal. This avoids changing coordinates/CRS while still achieving the intended
    physical spacing.

Multiprocessing robustness:
  - PDAL pipelines are executed via the `pdal pipeline` CLI inside each worker process.

Pipeline per input file:
  1) Reset Classification=0 and classify ground (Classification=2) via SMRF (default) or PMF.
  2) Rasterize ground-only points to a DEM GeoTIFF at a specified resolution (converted to dataset CRS units).
  3) Compute slope from the DEM (Horn 3x3 method) as degrees or percent (default).
  4) Compute roughness from the DEM as local elevation standard deviation in a moving window.

Output directory structure:
$out/
  DEM/
    <res_label>/
  L4_las/
  roughness/
    <res_label>/
  slope/
    <res_label>/

Roughness methods:
  - uniform (default): nodata-aware std-dev via weighted window means using scipy.ndimage.uniform_filter
  - generic: reference std-dev via scipy.ndimage.generic_filter

Requirements:
  - PDAL installed with `pdal` CLI available on PATH
  - rasterio, numpy
  - scipy (for roughness computation)

Example:
  python analyze_las.py --in path/to/input/data --out path/to/output/data --resolution 1usft --roughness-window 3usft --workers 8 --overwrite
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio

try:
    from scipy.ndimage import uniform_filter, generic_filter
except Exception:
    uniform_filter = None
    generic_filter = None


_US_SURVEY_FOOT_M = 1200.0 / 3937.0  # 0.3048006096012192


def _unit_name_to_unit_in_meters(name: str) -> float:
    """
    Map a unit name to meters-per-unit for the coordinate system.

    Args:
        name: One of {"m", "meter", "meters", "ft", "feet", "usft"}.

    Returns:
        Meters per 1 coordinate unit.
    """
    n = name.strip().lower()
    if n in {"m", "meter", "meters"}:
        return 1.0
    if n in {"ft", "foot", "feet"}:
        return 0.3048
    if n in {"usft", "us-ft", "surveyft", "survey-ft"}:
        return _US_SURVEY_FOOT_M
    raise ValueError(f"Unknown unit name: {name}")


@dataclass(frozen=True)
class ParsedDistance:
    """Distance parsed from a string, stored canonically in meters."""
    meters: float
    label: str  # folder-safe label


@dataclass(frozen=True)
class Paths:
    out_root: Path
    dem_dir: Path
    las_dir: Path
    rough_dir: Path
    slope_dir: Path


def parse_distance(s: str) -> ParsedDistance:
    """
    Parse a distance string like '10cm', '0.1m', '1ft', '3usft' into meters.

    Accepted units:
      - m, meter(s), metre(s)
      - cm
      - mm
      - ft, foot, feet (international foot)
      - usft, us-ft, surveyft, survey-ft (US survey foot)

    Returns:
        ParsedDistance: (meters, label)
    """
    raw = s.strip().lower().replace(" ", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-z\-]+)", raw)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid distance '{s}'. Examples: 0.1m, 10cm, 25mm, 1ft, 1usft"
        )

    value = float(m.group(1))
    unit = m.group(2)

    if value <= 0:
        raise argparse.ArgumentTypeError(f"Distance must be > 0, got '{s}'.")

    unit_norm = unit
    unit_norm = unit_norm.replace("metres", "m").replace("metre", "m")
    unit_norm = unit_norm.replace("meters", "m").replace("meter", "m")

    if unit_norm in {"m"}:
        meters = value
        label = f"{value:g}m"
    elif unit_norm in {"cm"}:
        meters = value / 100.0
        label = f"{value:g}cm"
    elif unit_norm in {"mm"}:
        meters = value / 1000.0
        label = f"{value:g}mm"
    elif unit_norm in {"ft", "foot", "feet"}:
        meters = value * 0.3048
        label = f"{value:g}ft"
    elif unit_norm in {"usft", "us-ft", "surveyft", "survey-ft"}:
        meters = value * _US_SURVEY_FOOT_M
        label = f"{value:g}usft"
    else:
        raise argparse.ArgumentTypeError(
            f"Unsupported unit in '{s}'. Use m, cm, mm, ft, or usft."
        )

    safe = re.sub(r"[^\w.-]+", "_", label).replace(".", "p")
    return ParsedDistance(meters=meters, label=safe)


def _ensure_dirs(out_dir: Path, res_label: str) -> Paths:
    dem_dir = out_dir / "DEM" / res_label
    las_dir = out_dir / "L4_las"
    rough_dir = out_dir / "roughness" / res_label
    slope_dir = out_dir / "slope" / res_label
    dem_dir.mkdir(parents=True, exist_ok=True)
    las_dir.mkdir(parents=True, exist_ok=True)
    rough_dir.mkdir(parents=True, exist_ok=True)
    slope_dir.mkdir(parents=True, exist_ok=True)
    return Paths(out_root=out_dir, dem_dir=dem_dir, las_dir=las_dir, rough_dir=rough_dir, slope_dir=slope_dir)


def _iter_inputs(in_path: Path) -> list[Path]:
    in_path = in_path.expanduser().resolve()
    if in_path.is_file():
        return [in_path]
    if not in_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {in_path}")
    exts = {".las", ".laz"}
    return sorted([p.resolve() for p in in_path.rglob("*") if p.suffix.lower() in exts])

_PDAL_INFO_JSON_FLAG: Optional[str] = None  # cached


def _detect_pdal_info_json_flag() -> Optional[str]:
    """
    Detect which flag (if any) makes `pdal info` output JSON.

    Returns:
        "--json" or "-j" if supported, else None.
    """
    global _PDAL_INFO_JSON_FLAG
    if _PDAL_INFO_JSON_FLAG is not None:
        return _PDAL_INFO_JSON_FLAG

    try:
        proc = subprocess.run(
            ["pdal", "info", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        _PDAL_INFO_JSON_FLAG = None
        return None

    # Common variants across PDAL versions
    if "--json" in help_text:
        _PDAL_INFO_JSON_FLAG = "--json"
    elif re.search(r"(^|\s)-j(\s|,|$)", help_text):
        _PDAL_INFO_JSON_FLAG = "-j"
    else:
        _PDAL_INFO_JSON_FLAG = None
    return _PDAL_INFO_JSON_FLAG


def _run_pdal_pipeline_cli(pipeline: list[object]) -> None:
    """Run a PDAL pipeline via the `pdal pipeline` CLI."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"pipeline": pipeline}, f)
        f.flush()
        pipeline_path = f.name

    try:
        subprocess.run(
            ["pdal", "pipeline", pipeline_path, "--verbose", "2"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = (
            f"PDAL CLI failed (exit={e.returncode}).\n"
            f"Pipeline JSON: {pipeline_path}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
        )
        raise RuntimeError(msg) from e
    finally:
        try:
            os.unlink(pipeline_path)
        except OSError:
            pass


def _pdal_info_json(path: Path) -> dict:
    """
    Return PDAL info metadata as a dict.

    Uses the older-compatible CLI form:
      pdal info --metadata -i <file>

    We intentionally do not request --summary here because some PDAL versions
    treat it as incompatible with --metadata.
    """
    cmd = ["pdal", "info", "--metadata", "-i", str(path)]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "PDAL failed to read metadata via `pdal info`.\n"
            f"Command: {cmd}\n"
            f"File: {path}\n"
            f"Exit code: {e.returncode}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
        ) from e

    out = (proc.stdout or "").strip()
    if not out:
        raise RuntimeError(f"`pdal info --metadata` returned empty output. Command: {cmd}")

    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Could not parse `pdal info --metadata` output as JSON.\n"
            f"Command: {cmd}\n"
            f"File: {path}\n"
            f"First 500 chars of output:\n{out[:500]}\n"
        ) from e


def _infer_linear_unit_meters_from_wkt_or_proj4(wkt: str, proj4: str) -> Optional[float]:
    """Infer CRS linear unit in meters from WKT/PROJ4 strings."""
    text = (wkt or "") + "\n" + (proj4 or "")
    t = text.lower()

    # PROJ4 hints
    if "+units=m" in t:
        return 1.0
    if "+units=ft" in t:
        return 0.3048
    if "+units=us-ft" in t or "+units=usft" in t:
        return _US_SURVEY_FOOT_M

    # WKT hints
    if 'unit["metre"' in t or 'unit["meter"' in t:
        return 1.0
    if "us survey foot" in t or "u.s. survey foot" in t:
        return _US_SURVEY_FOOT_M
    if 'unit["foot"' in t or 'unit["feet"' in t:
        return 0.3048

    return None


def _get_point_crs_unit_in_meters(in_las: Path) -> Optional[float]:
    """Determine the LAS CRS linear unit in meters, if possible; None if unknown or unreadable."""
    try:
        info = _pdal_info_json(in_las)
    except Exception:
        return None

    md = info.get("metadata", {}) or {}
    srs = md.get("srs", {}) or {}
    wkt = srs.get("wkt", "") or ""
    proj4 = srs.get("proj4", "") or ""
    return _infer_linear_unit_meters_from_wkt_or_proj4(wkt, proj4)



def _meters_to_crs_units(meters: float, unit_in_meters: Optional[float]) -> float:
    """
    Convert meters to CRS coordinate units.

    If unit_in_meters is None (unknown), assume meters (no conversion).
    """
    if unit_in_meters is None:
        return meters
    return meters / unit_in_meters


def classify_ground(
    in_las: Path,
    out_las: Path,
    method: str,
    smrf_window: float,
    smrf_slope: float,
    smrf_threshold: float,
    pmf_cell_size: float,
    pmf_max_window_size: float,
    pmf_slope: float,
) -> None:
    """Classify ground points and write a processed LAS/LAZ with updated Classification."""
    method = method.lower()
    if method not in {"smrf", "pmf"}:
        raise ValueError("method must be 'smrf' or 'pmf'")

    stages: list[object] = [
        str(in_las),
        {"type": "filters.assign", "value": "Classification = 0"},
    ]

    if method == "smrf":
        stages.append(
            {
                "type": "filters.smrf",
                "window": smrf_window,
                "slope": smrf_slope,
                "threshold": smrf_threshold,
            }
        )
    else:
        stages.append(
            {
                "type": "filters.pmf",
                "cell_size": pmf_cell_size,
                "max_window_size": pmf_max_window_size,
                "slope": pmf_slope,
            }
        )

    stages.append({"type": "writers.las", "filename": str(out_las)})
    _run_pdal_pipeline_cli(stages)


def generate_dem_from_ground(
    classified_las: Path,
    dem_tif: Path,
    resolution_in_crs_units: float,
    output_type: str = "mean",
) -> None:
    """Generate a DEM GeoTIFF from ground-only points using PDAL writers.gdal."""
    pipeline: list[object] = [
        str(classified_las),
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {
            "type": "writers.gdal",
            "filename": str(dem_tif),
            "resolution": float(resolution_in_crs_units),
            "output_type": output_type,
            "gdaldriver": "GTiff",
            "data_type": "float32",
            "nodata": -9999.0,
        },
    ]
    _run_pdal_pipeline_cli(pipeline)


def compute_slope(
    dem_tif: Path,
    slope_tif: Path,
    units: str = "degrees",
) -> None:
    """
    Compute slope from a DEM using Horn's 3x3 method.

    Args:
        dem_tif: Input DEM GeoTIFF.
        slope_tif: Output slope GeoTIFF.
        units: "degrees" or "percent".
    """
    units = units.lower()
    if units not in {"degrees", "percent"}:
        raise ValueError("slope units must be 'degrees' or 'percent'")

    with rasterio.open(dem_tif) as src:
        dem = src.read(1, masked=False).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

        dx = float(abs(src.transform.a))
        dy = float(abs(src.transform.e))
        if dx <= 0 or dy <= 0:
            raise RuntimeError("Invalid DEM geotransform; pixel size must be > 0.")

        invalid = np.zeros_like(dem, dtype=bool)
        if nodata is not None:
            invalid |= np.isclose(dem, float(nodata))
        invalid |= ~np.isfinite(dem)

        # Fill invalids with nearest-style edge replication via padding; this keeps the math stable.
        # We'll re-mask to nodata at the end.
        z = dem.copy()
        z[invalid] = np.nan

        # Pad by 1; use edge for boundaries, and if edge is nan, it remains nan.
        p = np.pad(z, pad_width=1, mode="edge")

        z1 = p[:-2, :-2]
        z2 = p[:-2, 1:-1]
        z3 = p[:-2, 2:]
        z4 = p[1:-1, :-2]
        z5 = p[1:-1, 1:-1]
        z6 = p[1:-1, 2:]
        z7 = p[2:, :-2]
        z8 = p[2:, 1:-1]
        z9 = p[2:, 2:]

        # Horn's method
        dzdx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * dx)
        dzdy = ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / (8.0 * dy)

        # Slope magnitude
        g = np.sqrt(dzdx * dzdx + dzdy * dzdy)

        if units == "degrees":
            slope = np.degrees(np.arctan(g)).astype(np.float32)
        else:
            slope = (g * 100.0).astype(np.float32)

        # Nodata where DEM invalid or where computation produced nan/inf
        out_invalid = invalid | ~np.isfinite(slope)
        slope_nodata = -9999.0
        slope[out_invalid] = slope_nodata

        profile.update(
            dtype="float32",
            nodata=slope_nodata,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

    with rasterio.open(slope_tif, "w", **profile) as dst:
        dst.write(slope, 1)


def compute_roughness(
    dem_tif: Path,
    rough_tif: Path,
    window_in_crs_units: float,
    method: str = "uniform",
) -> None:
    """
    Compute roughness from a DEM as local elevation standard deviation in a moving window.

    window_in_crs_units must be in the same linear units as the DEM pixel size (i.e., the LAS CRS units).
    """
    method = method.lower()
    if method not in {"uniform", "generic"}:
        raise ValueError("method must be 'uniform' or 'generic'")

    if method == "uniform" and uniform_filter is None:
        raise RuntimeError("SciPy is required for --roughness-method uniform (uniform_filter missing).")
    if method == "generic" and generic_filter is None:
        raise RuntimeError("SciPy is required for --roughness-method generic (generic_filter missing).")

    with rasterio.open(dem_tif) as src:
        dem = src.read(1, masked=False).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)

        px = max(1, int(round(window_in_crs_units / res_x)))
        py = max(1, int(round(window_in_crs_units / res_y)))
        if px % 2 == 0:
            px += 1
        if py % 2 == 0:
            py += 1

        invalid = np.zeros_like(dem, dtype=bool)
        if nodata is not None:
            invalid |= np.isclose(dem, float(nodata))
        invalid |= ~np.isfinite(dem)

        w = (~invalid).astype(np.float32)

        if method == "uniform":
            z = dem.copy()
            z[invalid] = 0.0
            z2 = z * z

            Ew = uniform_filter(w, size=(py, px), mode="nearest")
            Ez = uniform_filter(z * w, size=(py, px), mode="nearest")
            Ez2 = uniform_filter(z2 * w, size=(py, px), mode="nearest")

            eps = np.finfo(np.float32).eps
            denom = np.maximum(Ew, eps)

            mean = Ez / denom
            mean_sq = Ez2 / denom
            var = np.maximum(mean_sq - mean * mean, 0.0)
            rough = np.sqrt(var).astype(np.float32)

            no_samples = Ew <= eps
        else:
            dem_nan = dem.copy()
            dem_nan[invalid] = np.nan

            def _nanstd(values: np.ndarray) -> float:
                return float(np.nanstd(values, ddof=0))

            rough = generic_filter(
                dem_nan,
                function=_nanstd,
                size=(py, px),
                mode="nearest",
            ).astype(np.float32)

            no_samples = invalid

        rough_nodata = -9999.0
        rough[no_samples] = rough_nodata

        profile.update(
            dtype="float32",
            nodata=rough_nodata,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

    with rasterio.open(rough_tif, "w", **profile) as dst:
        dst.write(rough, 1)


def process_one(
    in_las: Path,
    paths: Paths,
    resolution_meters: float,
    method: str,
    window_meters: float,
    roughness_method: str,
    slope_units: str,
    input_units: Optional[str], 
    overwrite: bool,
    smrf_window: float,
    smrf_slope: float,
    smrf_threshold: float,
    pmf_cell_size: float,
    pmf_max_window_size: float,
    pmf_slope: float,
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single file end-to-end.

    Returns:
        (filename, success, error_message)
    """
    try:
        stem = in_las.stem
        out_las = paths.las_dir / f"{stem}.las"
        dem_tif = paths.dem_dir / f"{stem}.tif"
        slope_tif = paths.slope_dir / f"{stem}.tif"
        rough_tif = paths.rough_dir / f"{stem}.tif"

        targets = [out_las, dem_tif, slope_tif, rough_tif]
        if not overwrite and all(t.exists() for t in targets):
            return (str(in_las), True, None)

        if overwrite:
            for t in targets:
                if t.exists():
                    t.unlink()

        # Infer CRS units for this file and convert requested meters into CRS coordinate units.
        if input_units is not None:
            unit_in_m = _unit_name_to_unit_in_meters(input_units)
        else:
            unit_in_m = _get_point_crs_unit_in_meters(in_las)
            if unit_in_m is None:
                raise RuntimeError(
                    "Could not infer CRS linear units from LAS metadata, and --input-units was not provided.\n"
                    f"File: {in_las}\n"
                    "Fix: re-run with --input-units m|ft|usft."
                )
        resolution_in_crs_units = _meters_to_crs_units(resolution_meters, unit_in_m)
        window_in_crs_units = _meters_to_crs_units(window_meters, unit_in_m)

        classify_ground(
            in_las=in_las,
            out_las=out_las,
            method=method,
            smrf_window=smrf_window,
            smrf_slope=smrf_slope,
            smrf_threshold=smrf_threshold,
            pmf_cell_size=pmf_cell_size,
            pmf_max_window_size=pmf_max_window_size,
            pmf_slope=pmf_slope,
        )

        generate_dem_from_ground(
            classified_las=out_las,
            dem_tif=dem_tif,
            resolution_in_crs_units=resolution_in_crs_units,
            output_type="mean",
        )

        compute_slope(
            dem_tif=dem_tif,
            slope_tif=slope_tif,
            units=slope_units,
        )

        compute_roughness(
            dem_tif=dem_tif,
            rough_tif=rough_tif,
            window_in_crs_units=window_in_crs_units,
            method=roughness_method,
        )

        return (str(in_las), True, None)
    except Exception:
        return (str(in_las), False, traceback.format_exc())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Classify ground with PDAL, generate ground DEM, and compute roughness + slope rasters."
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="Input .las/.laz file or directory (recursively searched).",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory root.",
    )
    p.add_argument(
        "--resolution",
        dest="resolution",
        type=parse_distance,
        default=parse_distance("1usft"),
        help="Grid resolution with units (e.g., 0.1m, 10cm, 1ft, 1usft (default)).",
    )
    p.add_argument(
        "--roughness-window",
        dest="window",
        type=parse_distance,
        default=parse_distance("6usft"),
        help="Roughness window with units (e.g., 1m, 3ft). Default: 6usft.",
    )
    p.add_argument(
        "--roughness-method",
        choices=["uniform", "generic"],
        default="uniform",
        help="Roughness computation method (default: uniform).",
    )
    p.add_argument(
        "--slope-units",
        choices=["degrees", "percent"],
        default="percent",
        help="Slope output units (default: percent).",
    )
    p.add_argument(
        "--method",
        choices=["smrf", "pmf"],
        default="smrf",
        help="Ground classification method (default: smrf).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes. 0 = auto.",
    )
    p.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not force multiprocessing spawn context (not recommended on macOS).",
    )
    p.add_argument(
    "--input-units",
    choices=["m", "ft", "usft"],
    default=None,
    help=(
        "Linear units of the LAS coordinates, used only if CRS units cannot be inferred. "
        "Choices: m, ft, usft."
    ),
)

    # SMRF knobs
    p.add_argument("--smrf-window", type=float, default=16.0, help="SMRF window size.")
    p.add_argument("--smrf-slope", type=float, default=0.15, help="SMRF slope parameter.")
    p.add_argument("--smrf-threshold", type=float, default=0.5, help="SMRF elevation threshold.")

    # PMF knobs
    p.add_argument("--pmf-cell-size", type=float, default=1.0, help="PMF cell size.")
    p.add_argument("--pmf-max-window-size", type=float, default=33.0, help="PMF max window size.")
    p.add_argument("--pmf-slope", type=float, default=1.0, help="PMF slope parameter.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    # resolve output directory
    args.out_dir = args.out_dir.expanduser().resolve()

    # Ensure pdal CLI exists early
    try:
        subprocess.run(["pdal", "--version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise SystemExit(
            "Could not execute `pdal --version`. Ensure PDAL is installed and `pdal` is on PATH."
        ) from e

    in_files = _iter_inputs(args.in_path)
    if not in_files:
        raise SystemExit(f"No .las/.laz files found under: {args.in_path}")

    res_label = args.resolution.label
    paths = _ensure_dirs(args.out_dir, res_label)

    if args.workers and args.workers < 0:
        raise SystemExit("--workers must be >= 0")

    workers = args.workers if args.workers > 0 else min(32, (os.cpu_count() or 1))
    workers = max(1, workers)

    # Canonical meter values passed into workers; workers convert per-file to CRS units.
    resolution_meters = float(args.resolution.meters)
    window_meters = float(args.window.meters)

    results: list[Tuple[str, bool, Optional[str]]] = []

    if workers == 1 or len(in_files) == 1:
        for f in in_files:
            results.append(
                process_one(
                    in_las=f,
                    paths=paths,
                    resolution_meters=resolution_meters,
                    method=args.method,
                    window_meters=window_meters,
                    roughness_method=args.roughness_method,
                    slope_units=args.slope_units,
                    input_units=args.input_units,
                    overwrite=args.overwrite,
                    smrf_window=args.smrf_window,
                    smrf_slope=args.smrf_slope,
                    smrf_threshold=args.smrf_threshold,
                    pmf_cell_size=args.pmf_cell_size,
                    pmf_max_window_size=args.pmf_max_window_size,
                    pmf_slope=args.pmf_slope,
                )
            )
    else:
        ctx = None if args.no_spawn else mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {
                ex.submit(
                    process_one,
                    f,
                    paths,
                    resolution_meters,
                    args.method,
                    window_meters,
                    args.roughness_method,
                    args.slope_units,
                    args.input_units,
                    args.overwrite,
                    args.smrf_window,
                    args.smrf_slope,
                    args.smrf_threshold,
                    args.pmf_cell_size,
                    args.pmf_max_window_size,
                    args.pmf_slope,
                ): f
                for f in in_files
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    failed = [(fn, err) for (fn, ok, err) in results if not ok]
    if failed:
        print("\nSome files failed:\n")
        for fn, err in failed:
            print(f"--- {fn} ---")
            print(err)
        raise SystemExit(f"{len(failed)} of {len(results)} files failed.")
    else:
        print(f"Done. Processed {len(results)} file(s) with workers={workers}.")


if __name__ == "__main__":
    main()
