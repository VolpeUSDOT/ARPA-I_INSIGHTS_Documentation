#!/usr/bin/env python3
"""Stage 2 -- point-cloud noise measurement from sampled tiles
Stage 2 of the noise characterization.

Downloads a seeded sample of *interior* tiles per sortie from S3, computes a
per-tile metric record, and appends it to a JSONL checkpoint file.  Every tile is
checkpointed as it completes and already-measured ids are skipped on restart, so
an interrupted run does not throw away its downloads.

The measurement has two independent detectors, and BOTH are required:

  * an ISOLATION test (section 3.3) which finds sparse floating returns without
    reference to height, so real buildings survive; and
  * a plan-view AREAL DENSITY test (section 3.4) which finds dense banded
    artifact layers that the isolation test cannot see.

Use stage2_report.py to turn the JSONL into the per-sortie table, and
stage2_checks.py for the validation checks of section 3.8.

Run with --help for options.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- #
# Constants (section 3).  These are the published parameters; do not tune them
# to make a result come out.
# ---------------------------------------------------------------------------- #
M_PER_FTUS = 0.3048006096012192   # both projected CRSs are in US survey FEET
CELL_M = 5.0                      # plan-view ground/occupancy grid
GROUND_PCT = 5.0                  # percentile of Z used as the ground surface
GROUND_FILL_PASSES = 6            # 3x3 neighbour-mean hole-fill passes
ISO_VOX_M = 2.0                   # isolation voxel edge
ISO_MIN_PTS = 20                  # < this many points in the 3x3x3 neighbourhood
HIGH_HAG_M = 20.0                 # high-noise height-above-ground threshold
LOW_HAG_M = -2.0                  # low-noise threshold; the release's vertical
                                  # registration RMSE is 0.36 m, so -2 m is well
                                  # outside registration error
BAND_BINS_M = 10.0                # histogram bin width for vertical structure
ABOVE_THRESHOLDS_M = (20, 50, 100, 150, 200, 300)
BELOW_THRESHOLDS_M = (2, 5, 10, 25, 50)
DENSITY_THRESHOLDS_M = (20, 50)   # areal-density / band-structure thresholds
DEFAULT_SEED = 20260810
DEFAULT_N = 30

BUCKET = "arpa-i-insights"
S3_PREFIX = f"s3://{BUCKET}/"
INDEX_RELPATH = Path("data/lidar/v1/stac/index/items.parquet")
TILE_ID_RE = re.compile(r"_X(\d+)_Y(\d+)")

# Projected CRS per sortie.  Both are US survey feet.
EPSG_BY_SORTIE = {
    "DRCOG": 6430, "FrontRange": 6430, "I25N": 6430, "I25S2": 6430,
    "I70A": 6430, "I70BC": 6430, "I70D": 6430, "I70E": 6430, "I70F": 6430,
    "I70G": 6430, "I70H": 6430, "I70I": 6430,
    "I15South": 6626, "I80East": 6626, "I80P1": 6626, "I80P2": 6626,
    "SLC": 6626,
}


# ---------------------------------------------------------------------------- #
# Index handling
# ---------------------------------------------------------------------------- #
def default_index_path() -> Path:
    """Resolve the index relative to the repo root, never to a scratch path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / INDEX_RELPATH
        if candidate.exists():
            return candidate
    return here.parents[2] / INDEX_RELPATH


def load_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        index_path,
        columns=["id", "sortie", "pc:count", "proj:epsg", "proj:bbox", "bbox",
                 "elevation:min", "elevation:max", "elevation:median", "s3_path"],
    )
    gx = np.empty(len(df), dtype=np.int32)
    gy = np.empty(len(df), dtype=np.int32)
    for i, tid in enumerate(df["id"].to_numpy()):
        m = TILE_ID_RE.search(tid)
        if m is None:
            raise ValueError(f"cannot parse grid indices from id {tid!r}")
        gx[i], gy[i] = int(m.group(1)), int(m.group(2))
    df["gx"], df["gy"] = gx, gy
    # TRAP: proj:bbox and bbox are JSON strings, not lists.
    pb = np.array([json.loads(s) for s in df["proj:bbox"]])
    df["minx"], df["miny"], df["maxx"], df["maxy"] = pb[:, 0], pb[:, 1], pb[:, 2], pb[:, 3]
    bb = np.array([json.loads(s) for s in df["bbox"]])
    df["lon"] = (bb[:, 0] + bb[:, 2]) / 2
    df["lat"] = (bb[:, 1] + bb[:, 3]) / 2
    return df


def interior_ids(sub: pd.DataFrame) -> list[str]:
    """Tiles all eight of whose grid neighbours exist in the same sortie.

    Edge tiles may fill only part of their footprint, which corrupts both the
    ground surface (a clipped tile has fewer cells to interpolate across) and
    every rate denominator built from cell counts.  This is the same interior
    definition the manuscript already uses for its point-density validation.
    """
    present = set(zip(sub["gx"].tolist(), sub["gy"].tolist()))
    out = []
    for tid, gx, gy in zip(sub["id"].tolist(), sub["gx"].tolist(), sub["gy"].tolist()):
        if all((gx + dx, gy + dy) in present
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if not (dx == 0 and dy == 0)):
            out.append(tid)
    return out


def sample_sortie(sub: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Draw n interior tiles from one sortie, reproducibly.

    TRAP: the per-sortie seed uses zlib.crc32 of the sortie name, NEVER the
    built-in hash().  PYTHONHASHSEED randomises str hashing per process, so a
    hash()-based seed draws a different sample on every invocation -- which
    destroys both reproducibility and the JSONL checkpoint/resume logic (the
    resume would skip ids the new run never intended to draw).  crc32 is a fixed
    function of the bytes.
    """
    sortie = str(sub["sortie"].iloc[0])
    ids = sorted(interior_ids(sub))          # sort so the pool order is fixed too
    if not ids:
        return []
    rng = np.random.default_rng(zlib.crc32(sortie.encode()) + seed)
    k = min(n, len(ids))
    pick = rng.choice(len(ids), size=k, replace=False)
    return [ids[i] for i in pick]


def build_sample(df: pd.DataFrame, n: int, seed: int,
                 sorties: list[str] | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sortie, sub in df.groupby("sortie", sort=True):
        if sorties and sortie not in sorties:
            continue
        out[sortie] = sample_sortie(sub, n, seed)
    return out


def assert_sample_reproducible(index: Path, n: int, seed: int) -> None:
    """Run the sampler in two subprocesses with different PYTHONHASHSEED values
    and require identical output.  This is the regression test for the
    crc32-not-hash() trap above."""
    outs = []
    for hs in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--print-sample",
             "--index", str(index), "--n", str(n), "--seed", str(seed)],
            capture_output=True, text=True, env=env, check=True)
        outs.append(r.stdout)
    if outs[0] != outs[1]:
        raise AssertionError(
            "sample is NOT reproducible across processes -- a hash()-based seed "
            "has been reintroduced somewhere")
    print(f"sample reproducibility across PYTHONHASHSEED=0/1: OK "
          f"({len(outs[0].splitlines())} tiles)")


# ---------------------------------------------------------------------------- #
# Point fetch
# ---------------------------------------------------------------------------- #
def make_s3(workers: int):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=max(2 * workers, 32),
        retries={"max_attempts": 5, "mode": "standard"},
    ))


def fetch_xyz(s3, s3_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download one COPC LAZ tile and return (x, y, z) as float64 in the tile's
    native projected units (US survey feet).

    MEMORY DISCIPLINE (mandatory): an earlier 16-worker run was OOM-killed
    because laspy objects were held alive across threads.  Copy the three
    ordinates out as plain float64 arrays, then drop the LasData object and close
    the buffer immediately so the ~12 MB compressed body and its decoded point
    record are not pinned while the metrics run.
    """
    import laspy
    key = s3_path[len(S3_PREFIX):] if s3_path.startswith(S3_PREFIX) else s3_path
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    buf = io.BytesIO(body)
    las = laspy.read(buf)
    x = np.asarray(las.x, dtype=np.float64).copy()
    y = np.asarray(las.y, dtype=np.float64).copy()
    z = np.asarray(las.z, dtype=np.float64).copy()
    del las
    buf.close()
    del buf, body
    return x, y, z


# ---------------------------------------------------------------------------- #
# Core per-tile computation
# ---------------------------------------------------------------------------- #
def _box3_sum(a: np.ndarray) -> np.ndarray:
    """Sum over the 3x3 neighbourhood (zero-padded), for the hole-fill."""
    p = np.pad(a, 1, mode="constant", constant_values=0.0)
    out = np.zeros_like(a)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            out += p[dx:dx + a.shape[0], dy:dy + a.shape[1]]
    return out


def plan_grid(x: np.ndarray, y: np.ndarray, cell_m: float = CELL_M):
    """Build the plan-view cell index for the tile.

    TRAP: nx = max(ceil(span / cell), 1) with NO "+1".  Adding one gives a
    trailing row and column that no point can fall into, which inflates
    empty_cell_frac and sparse_cell_frac and silently manufactures "voids".
    Indices are CLIPPED to nx-1 so the point sitting exactly on x.max() lands in
    the last real cell rather than in a phantom one.
    """
    cell = cell_m / M_PER_FTUS          # the coordinates are in US survey feet
    x0, y0 = x.min(), y.min()
    nx = max(int(math.ceil((x.max() - x0) / cell)), 1)
    ny = max(int(math.ceil((y.max() - y0) / cell)), 1)
    ix = np.clip(((x - x0) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - y0) / cell).astype(np.int64), 0, ny - 1)
    return ix, iy, nx, ny, ix * ny + iy


def ground_surface(cid: np.ndarray, z: np.ndarray, ncells: int,
                   pct: float = GROUND_PCT,
                   passes: int = GROUND_FILL_PASSES,
                   shape: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
    """Per-cell ground elevation = `pct`-th percentile of Z, holes filled.

    A low PERCENTILE rather than the minimum: sparse below-surface noise would
    otherwise drag the ground surface down under the noise itself, which both
    hides low noise and inflates the apparent height of everything above.
    """
    order = np.lexsort((z, cid))
    cs, zs = cid[order], z[order]
    uniq, start, cnt = np.unique(cs, return_index=True, return_counts=True)
    # Linear-interpolated percentile within each cell's sorted Z run.
    pos = start + (pct / 100.0) * (cnt - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    frac = pos - lo
    gvals = zs[lo] * (1.0 - frac) + zs[hi] * frac

    ground = np.full(ncells, np.nan)
    ground[uniq] = gvals
    nx, ny = shape
    g = ground.reshape(nx, ny)

    # Iterative 3x3 neighbour-mean hole-fill.
    n_unfilled = 0
    for _ in range(passes):
        holes = np.isnan(g)
        if not holes.any():
            break
        valid = (~holes).astype(np.float64)
        ssum = _box3_sum(np.where(holes, 0.0, g))
        scnt = _box3_sum(valid)
        fillable = holes & (scnt > 0)
        g = np.where(fillable, ssum / np.maximum(scnt, 1.0), g)
    holes = np.isnan(g)
    if holes.any():
        # Fall back to the tile-wide ground level for any cell still isolated
        # from data after `passes` passes; recorded so it is visible.
        n_unfilled = int(holes.sum())
        g = np.where(holes, np.percentile(z, pct), g)
    return g, n_unfilled


def isolated_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  vox_m: float = ISO_VOX_M,
                  min_pts: int = ISO_MIN_PTS) -> np.ndarray:
    """True where the point's 3x3x3 voxel neighbourhood holds < min_pts points.

    The release is sampled at ~87 points/m^2, so any real surface fills a 2 m
    voxel neighbourhood far beyond 20 points; sparse floating returns do not.
    The test makes no reference to height, which is exactly why tall buildings
    survive it (see stage2_checks.py negative control).

    Implementation: voxel-key + np.unique + searchsorted over the 27 shifted
    keys.  A dense 3D array would be gigabytes for a mountainous tile.
    """
    inv_v = 1.0 / vox_m
    # Work in metres so the voxel is actually 2 m on a side.
    vx = np.floor(x * M_PER_FTUS * inv_v).astype(np.int64)
    vy = np.floor(y * M_PER_FTUS * inv_v).astype(np.int64)
    vz = np.floor(z * M_PER_FTUS * inv_v).astype(np.int64)
    # Shift to a 1-based origin and size the dimensions with one voxel of slack
    # at each end, so the +/-1 key offsets below cannot wrap into a neighbouring
    # row of the linearised index.
    vx -= vx.min() - 1
    vy -= vy.min() - 1
    vz -= vz.min() - 1
    ny_v = int(vy.max()) + 2
    nz_v = int(vz.max()) + 2
    key = (vx * ny_v + vy) * nz_v + vz
    del vx, vy, vz

    uk, inv, cts = np.unique(key, return_inverse=True, return_counts=True)
    del key
    nbr = np.zeros(uk.shape[0], dtype=np.int64)
    last = uk.shape[0] - 1
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                shifted = uk + (dx * ny_v + dy) * nz_v + dz
                pos = np.searchsorted(uk, shifted)
                np.clip(pos, 0, last, out=pos)
                hit = uk[pos] == shifted
                nbr[hit] += cts[pos[hit]]
    return nbr[inv] < min_pts


def _cellset_stats(cid_sel: np.ndarray, n_sel: int, n_cells_occupied: int,
                   median_cell_count: float, prefix: str) -> dict:
    """cell_frac / pts_per_affected_cell / areal density / density ratio."""
    if n_sel == 0:
        return {f"{prefix}_cell_frac": 0.0,
                f"{prefix}_pts_per_affected_cell": 0.0,
                f"{prefix}_areal_density_m2": 0.0,
                f"{prefix}_density_ratio": 0.0,
                f"{prefix}_n_cells": 0}
    cells = np.unique(cid_sel)
    ppc = n_sel / len(cells)
    return {
        f"{prefix}_cell_frac": len(cells) / n_cells_occupied,
        f"{prefix}_pts_per_affected_cell": float(ppc),
        f"{prefix}_areal_density_m2": float(ppc / (CELL_M ** 2)),
        f"{prefix}_density_ratio": float(ppc / median_cell_count),
        f"{prefix}_n_cells": int(len(cells)),
    }


def _vertical_structure(h: np.ndarray, prefix: str) -> dict:
    """Vertical structure of a height population: 10 m histogram peak, quantiles,
    IQR and Shannon entropy (bits) over the bins.

    Low entropy + narrow IQR = a banded layer.  High entropy + wide IQR = a
    diffuse population (vegetation, genuine structure).
    """
    out = {f"{prefix}_n": int(h.size)}
    if h.size == 0:
        out.update({f"{prefix}_peak_m": None, f"{prefix}_p10": None,
                    f"{prefix}_p25": None, f"{prefix}_p50": None,
                    f"{prefix}_p75": None, f"{prefix}_p90": None,
                    f"{prefix}_iqr": None, f"{prefix}_entropy_bits": None})
        return out
    lo = math.floor(h.min() / BAND_BINS_M) * BAND_BINS_M
    hi = math.ceil(h.max() / BAND_BINS_M) * BAND_BINS_M + BAND_BINS_M
    bins = np.arange(lo, hi + BAND_BINS_M / 2, BAND_BINS_M)
    counts, edges = np.histogram(h, bins=bins)
    p = counts[counts > 0] / counts.sum()
    k = int(np.argmax(counts))
    q10, q25, q50, q75, q90 = np.percentile(h, [10, 25, 50, 75, 90])
    out.update({
        f"{prefix}_peak_m": float((edges[k] + edges[k + 1]) / 2),
        f"{prefix}_p10": float(q10), f"{prefix}_p25": float(q25),
        f"{prefix}_p50": float(q50), f"{prefix}_p75": float(q75),
        f"{prefix}_p90": float(q90), f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_entropy_bits": float(-np.sum(p * np.log2(p))),
    })
    return out


def tile_metrics(x: np.ndarray, y: np.ndarray, z: np.ndarray, *,
                 cell_m: float = CELL_M,
                 ground_pct: float = GROUND_PCT,
                 iso_vox_m: float = ISO_VOX_M,
                 iso_min_pts: int = ISO_MIN_PTS) -> dict:
    """All per-tile metrics of sections 3.2-3.5 for one tile's points."""
    n = x.size
    ix, iy, nx, ny, cid = plan_grid(x, y, cell_m)
    ncells = nx * ny

    counts = np.bincount(cid, minlength=ncells)
    occ = counts > 0
    n_cells_occupied = int(occ.sum())
    median_cell_count = float(np.median(counts[occ]))

    ground, n_unfilled = ground_surface(cid, z, ncells, pct=ground_pct,
                                        shape=(nx, ny))
    # Height above ground, in METRES.  Working in HAG rather than raw elevation
    # is essential: within-tile elevation range is dominated by real terrain
    # slope in the mountainous sorties, so a raw-elevation threshold would flag
    # hillsides as noise.
    hag = (z - ground[ix, iy]) * M_PER_FTUS

    iso = isolated_mask(x, y, z, vox_m=iso_vox_m, min_pts=iso_min_pts)

    rec: dict = {
        "n_points": int(n),
        "nx": nx, "ny": ny,
        "n_cells_total": int(ncells),
        "n_cells_occupied": n_cells_occupied,
        "median_cell_count": median_cell_count,
        "empty_cell_frac": float(1.0 - n_cells_occupied / ncells),
        # Empty cells have count 0, which is already < 10% of the median, so this
        # single comparison is "sparse cells plus empty cells".
        "sparse_cell_frac": float((counts < 0.10 * median_cell_count).sum() / ncells),
        "ground_unfilled_cells": n_unfilled,
        "hag_max_m": float(hag.max()),
        "hag_min_m": float(hag.min()),
        "n_isolated": int(iso.sum()),
        "iso_frac": float(iso.mean()),
    }

    # --- band profile: NOT isolation filtered -------------------------------
    for t in ABOVE_THRESHOLDS_M:
        rec[f"frac_above_{t}m"] = float(np.count_nonzero(hag > t) / n)
    for t in BELOW_THRESHOLDS_M:
        rec[f"frac_below_{t}m"] = float(np.count_nonzero(hag < -t) / n)

    # --- areal-density test (section 3.4): the isolation filter is DELIBERATELY
    #     NOT applied here.  That is the whole point: in I70E 9-16% of points lie
    #     above 20 m AGL but only 1-2% of them are isolated, because the banded
    #     artifact layer is locally DENSE.  Filtering by isolation first would
    #     discard the very population this metric is meant to characterise.
    for t in DENSITY_THRESHOLDS_M:
        sel = hag > t
        rec.update(_cellset_stats(cid[sel], int(sel.sum()), n_cells_occupied,
                                  median_cell_count, f"el{t}"))
        rec[f"n_above_{t}m"] = int(sel.sum())

    # --- isolated populations -----------------------------------------------
    hi = (hag > HIGH_HAG_M) & iso
    lo = (hag < LOW_HAG_M) & iso
    rec["n_high_noise"] = int(hi.sum())
    rec["n_low_noise"] = int(lo.sum())
    rec.update(_cellset_stats(cid[hi], int(hi.sum()), n_cells_occupied,
                              median_cell_count, "hi"))
    rec.update(_cellset_stats(cid[lo], int(lo.sum()), n_cells_occupied,
                              median_cell_count, "lo"))
    hi_cells = set(np.unique(cid[hi]).tolist())
    lo_cells = set(np.unique(cid[lo]).tolist())
    rec["hi_cells_also_low_frac"] = (
        len(hi_cells & lo_cells) / len(hi_cells) if hi_cells else 0.0)

    # Fraction of the ELEVATED population that the isolation test flags.  This
    # is the number that exposes the isolation test's blind spot on dense bands
    # (section 3.8 check 2) and that vindicates it on real highrises (check 1).
    for t in DENSITY_THRESHOLDS_M:
        n_el = rec[f"n_above_{t}m"]
        n_el_iso = int(np.count_nonzero((hag > t) & iso))
        rec[f"n_above_{t}m_isolated"] = n_el_iso
        rec[f"iso_share_of_above_{t}m"] = (n_el_iso / n_el) if n_el else 0.0

    # --- vertical structure, ISOLATED POINTS ONLY ---------------------------
    # MUST exclude non-isolated points: otherwise vegetation dominates the
    # histogram in the clean sorties and produces a spurious "band" at treetop
    # height.
    for t in DENSITY_THRESHOLDS_M:
        rec.update(_vertical_structure(hag[(hag > t) & iso], f"iso{t}"))

    return rec


# ---------------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------------- #
def measure_tiles(index: pd.DataFrame, tile_ids: list[str], out_path: Path,
                  workers: int, extra: dict | None = None,
                  metric_kwargs: dict | None = None,
                  resume: bool = True) -> int:
    """Measure `tile_ids`, appending one JSON object per line to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume and out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001 - tolerate a torn final line
                    pass
        if done:
            print(f"resume: {len(done)} tiles already in {out_path}")

    todo = [t for t in tile_ids if t not in done]
    if not todo:
        print("nothing to do")
        return 0
    meta = index.set_index("id").loc[todo]
    s3 = make_s3(workers)
    lock = threading.Lock()
    fh = out_path.open("a")
    t0 = time.time()
    counter = [0]
    n_fail = [0]

    def work(tid: str) -> dict:
        row = meta.loc[tid]
        x, y, z = fetch_xyz(s3, row["s3_path"])
        rec = tile_metrics(x, y, z, **(metric_kwargs or {}))
        del x, y, z
        rec["id"] = tid
        rec["sortie"] = row["sortie"]
        rec["gx"] = int(row["gx"])
        rec["gy"] = int(row["gy"])
        rec["lon"] = float(row["lon"])
        rec["lat"] = float(row["lat"])
        rec["index_pc_count"] = int(row["pc:count"])
        if extra:
            rec.update(extra)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, t): t for t in todo}
        for f in as_completed(futs):
            tid = futs[f]
            try:
                rec = f.result()
            except Exception as exc:  # noqa: BLE001 - one bad tile must not kill the run
                with lock:
                    n_fail[0] += 1
                    print(f"  FAIL {tid}: {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()          # checkpoint every tile, not every buffer
                os.fsync(fh.fileno())
                counter[0] += 1
                el = time.time() - t0
                print(f"  [{counter[0]}/{len(todo)}] {tid} "
                      f"n={rec['n_points']} hi={rec['n_high_noise']} "
                      f"lo={rec['n_low_noise']} ({el:.0f}s)", flush=True)
    fh.close()
    print(f"measured {counter[0]} tiles ({n_fail[0]} failures) in "
          f"{time.time() - t0:.0f}s -> {out_path}")
    return n_fail[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2: sample interior tiles and measure per-tile noise "
                    "metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--index", type=Path, default=default_index_path(),
                    help="path to items.parquet")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="interior tiles sampled per sortie")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="added to crc32(sortie) to seed the per-sortie RNG")
    ap.add_argument("--workers", type=int, default=16,
                    help="download/compute threads (network-bound; 16-24 is the "
                         "useful range, and higher risks memory pressure)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "out" / "stage2_tiles.jsonl",
                    help="JSONL checkpoint file (appended to; ids already "
                         "present are skipped)")
    ap.add_argument("--sorties", nargs="*", default=None,
                    help="restrict to these sorties (default: all 17)")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not skip ids already present in --out")
    ap.add_argument("--print-sample", action="store_true",
                    help="print the sampled tile ids and exit (no downloads)")
    ap.add_argument("--check-sample", action="store_true",
                    help="assert sample reproducibility across two subprocesses "
                         "with different PYTHONHASHSEED, then continue")
    args = ap.parse_args(argv)

    if not args.index.exists():
        ap.error(f"index not found: {args.index}")

    df = load_index(args.index)
    sample = build_sample(df, args.n, args.seed, args.sorties)

    if args.print_sample:
        for sortie in sorted(sample):
            for tid in sample[sortie]:
                print(tid)
        return 0

    if args.check_sample:
        assert_sample_reproducible(args.index, args.n, args.seed)

    tile_ids = [t for s in sorted(sample) for t in sample[s]]
    print(f"index: {args.index}")
    print(f"sample: {len(tile_ids)} interior tiles across {len(sample)} sorties "
          f"(n={args.n}, seed={args.seed})")
    for sortie in sorted(sample):
        print(f"  {sortie:<11s} {len(sample[sortie])}")
    n_fail = measure_tiles(df, tile_ids, args.out, args.workers,
                           resume=not args.no_resume)
    print("\nnext: stage2_report.py --tiles", args.out)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
