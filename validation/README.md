# INSIGHTS Technical Validation — reference scripts

Reconstruction of the analyses reported in the *Technical Validation* section of the
manuscript (`insights-writeup/main.tex`): the point-cloud noise characterization, the
DRCOG coverage-void material, and the release-wide area computation.

Each script documents its own method in its module docstring: the algorithm, the
parameter values, and the reasoning behind each choice. The *Traps* section below indexes
the mistakes that caused real errors while this analysis was first developed, and each is
also commented at the point of use in the code. Read that section before modifying
anything or quoting a number.

---

## Setup

`pdal` is listed in the project `pyproject.toml`, but its Python bindings cannot build
here (no PDAL C++ library present), so `uv run` against the project fails. Build a
standalone environment instead — none of these scripts need PDAL:

```sh
cd <repo root>
uv venv .venv-validation --python 3.11
uv pip install --python .venv-validation/bin/python \
    pandas pyarrow numpy laspy lazrs shapely boto3 pyproj scipy
```

* `lazrs` is required to decode COPC LAZ through `laspy`.
* `scipy` provides the Spearman correlation (Stage 1) and the max/min filters.
* `matplotlib` is not needed; no script produces figures.

All scripts resolve `data/lidar/v1/stac/index/items.parquet` relative to the repo root
and accept `--index` to override it. Nothing writes to `/tmp`; the default output
directory is `validation/out/`. S3 access is anonymous (bucket `arpa-i-insights`).

Throughput is **network-bound**: about 12 MB and a few seconds per tile at concurrency,
against ~0.2 s of compute. 16–24 worker threads is the useful range; more buys nothing
and risks memory pressure. If a run stalls, *reduce* `--workers` and restart — the
tile-sampling stages checkpoint every tile to JSONL and skip already-measured ids.

---

## What each script establishes

| script | network | establishes |
|---|---|---|
| `noise/stage1_index_screen.py` | no | **negative**: index elevation excursion tracks terrain, not noise |
| `noise/stage2_noise_sample.py` | yes | per-tile noise metrics for a seeded sample of interior tiles |
| `noise/stage2_report.py` | no | the per-sortie noise table |
| `noise/stage2_checks.py` | yes | validation checks, including the negative control |
| `coverage/stage3_drcog_void.py` | `--mode headers` only | the DRCOG coverage void, and that `proj:bbox` describes the data |
| `coverage/area_recompute.py` | no | 1610.50 km² footprint area and 87.00 points/m² |
| `georegistration/characterize_offsets.py` | yes | inter-sortie vertical offsets, and whether one bias per sortie explains them |
| `georegistration/test_relative_accuracy.py` | yes | first screen of relative vs absolute accuracy over co-observed ground |
| `georegistration/verify_offset.py` | yes | that an offset is a constant shift, not an estimator artifact or a warp |

### `georegistration/` — inter-sortie vertical agreement

Because sorties overlap, the same ground is sometimes observed twice, which allows the two
independent looks to be differenced directly. These scripts do that over flat, hard,
fully sampled ground: a 2 m grid across the overlap, a per-cell ground proxy taken as the
10th percentile of elevation, and a cell kept only where both sorties have at least 30
points and a within-cell spread under 0.5 m.

The distinction they draw is between *shape* agreement and *height* agreement. Within any
one overlap the two surfaces are nearly parallel, agreeing to a few centimetres, which is
far better than the absolute georegistration RMSE. Between sorties, however, they are
offset vertically by metre order.

`characterize_offsets.py` measures every overlapping sortie pair and then solves, per
connected component of the overlap graph, for a single vertical bias per sortie
(`d_ij = b_i - b_j`, gauged so the biases sum to zero). Components with more edges than
nodes minus one are over-determined, so the residuals test whether the offsets really are
a per-sortie property — that is, whether a per-sortie vertical correction exists at all.
It excludes the western third of `I80P2`, where dense noise would corrupt a ground
estimate.

### Positive and negative results — read this before quoting anything

**Positive results** (the analysis supports the claim):

* **The noise characterization itself.** Sorties differ by three orders of
  magnitude in high-noise rate, and I70E is dramatically the worst on every measure.
* **The areal-density discriminator.** Artifact layers deposit a few points per
  m² (density ratio ≈ 0.004–0.158); real roofs and canopy are sampled at close to the
  tile's own density (ratio ≈ 0.60–1.50). The two populations do not overlap, the
  nearest values being about a factor of 4 apart. This is the load-bearing result of the
  whole noise section, and `stage2_checks.py negative-control` is what demonstrates it.
* **The DRCOG coverage void.** A data gap of median width 110.7 m is present in
  **all 246** rows where both grid columns exist, centred at 105°00'34.3"W, running the
  full ~34 km N/S extent of the sortie. It is missing **area**, not reduced density:
  density within the occupied part of the affected column matches the rest of the sortie
  to within 0.2%.
* **The LAS header cross-check.** Across all 36,221 DRCOG tiles the LAS public
  header bounding box reproduces `proj:bbox` exactly and the header point counts match
  the index for every tile.
* **The release-wide area.** 1610.50 km² by footprint-polygon sum, agreeing to
  1.000000 between equal-area, native State Plane and geodesic integration.

**Negative results** (the analysis does *not* support a claim, and the manuscript
correctly makes none):

* **Stage 1 relief confound.** The elevation-excursion metric correlates with local
  terrain relief, and the *strongest* correlation in the release (FrontRange, ρ = 0.747)
  belongs to a **clean** sortie. The metric is therefore tracking real terrain slope,
  not noise, and a large excursion is not evidence of an artifact. Absolute elevation
  shows no consistent relationship either. **The manuscript deliberately makes no relief
  claim on the strength of this stage.** Noise has to be measured in height above ground
  on the actual points, which is what Stage 2 does.
* **DRCOG water association not confirmed** (`stage2_checks.py --check drcog-water`). Comparing DRCOG interior
  tiles from the lowest point-count decile against density-matched controls gives the
  *same* proportion of noise-bearing tiles. The QC observation that DRCOG's sparse high
  noise sits over water is **not** independently confirmed by this test; the manuscript
  reports it as a visual-inspection observation, and it must stay that way.

---

## Running

Everything below assumes `PY=.venv-validation/bin/python` from the repo root.

### Index-only (cheap, and exact — these reproduce the published figures)

```sh
$PY validation/noise/stage1_index_screen.py
$PY validation/coverage/area_recompute.py
$PY validation/coverage/stage3_drcog_void.py --mode bbox
```

Each runs in a couple of seconds over all 82,429 index rows.

### LAS header screen (network, ~375 bytes per tile, ~10 min for all of DRCOG)

```sh
$PY validation/coverage/stage3_drcog_void.py --mode headers --workers 32
```

Add `--limit 500` for a quick sanity pass.

### Tile sampling (network, ~12 MB per tile)

```sh
# full reproduction: 17 sorties x 30 interior tiles = 510 tiles
$PY validation/noise/stage2_noise_sample.py --n 30 --workers 16 --check-sample \
    --out validation/out/stage2_tiles.jsonl
$PY validation/noise/stage2_report.py --tiles validation/out/stage2_tiles.jsonl

# smoke test: 5 tiles per sortie
$PY validation/noise/stage2_noise_sample.py --n 5 --workers 16 \
    --out validation/out/stage2_tiles_n5.jsonl
```

`--check-sample` re-runs the sampler in two subprocesses with different
`PYTHONHASHSEED` values and asserts identical output. `--print-sample` lists the drawn
tile ids without downloading anything.

A reduced `--n` will **not** reproduce the published per-sortie table and should not be tuned to make it
do so; `stage2_report.py` prints a warning when it sees `n < 30`.

### Validation checks

```sh
$PY validation/noise/stage2_checks.py negative-control   # 4 downloads — run this one
$PY validation/noise/stage2_checks.py iso-sweep
$PY validation/noise/stage2_checks.py ground-sweep
$PY validation/noise/stage2_checks.py i80p2-thirds
$PY validation/noise/stage2_checks.py drcog-water        # 50 downloads
$PY validation/noise/stage2_checks.py all
```

`negative-control` is the single most important check in the method: it is what shows
real buildings are not counted as noise.

---

## Traps, and where they are guarded

Each of these caused a real error in the original work. Every one is commented at the
point of use in the code; this is the index.

| trap | why it matters | guarded in |
|---|---|---|
| `proj:bbox` / `bbox` are JSON **strings** | silent `TypeError` or per-character iteration | `load_index`, `load_drcog` |
| `geometry` ring has **6 XYZ** coords (4 corners + closing vertex + its duplicate) | corrupts every polygon area | `load_corners`, `load_drcog` |
| projected CRSs are **US survey feet** | `M_PER_FTUS = 0.3048006096012192`, not the international foot | all scripts |
| `crc32`, never built-in `hash()` | `PYTHONHASHSEED` randomizes `str` hashing per process, so a `hash()` seed redraws the sample every run and breaks checkpoint resume | `sample_sortie`, asserted by `--check-sample` |
| **no `+1`** on the plan grid | a phantom trailing row/column can hold no point, inflating `empty_cell_frac` and manufacturing "voids" | `plan_grid` |
| ground = low **percentile**, not minimum | the minimum lets sparse below-surface noise drag the surface down under the noise itself | `ground_surface` |
| work in **height above ground**, not raw elevation | within-tile elevation range is dominated by real terrain slope in the mountainous sorties | `tile_metrics` |
| areal-density metrics must **exclude** the isolation filter | the artifact layer is locally *dense*; filtering by isolation first discards the population being characterised | `tile_metrics`, areal-density block |
| vertical structure must **include only** isolated points | otherwise vegetation dominates the histogram in clean sorties and fakes a treetop-height "band" | `_vertical_structure` call sites |
| footprint polygon ≠ `proj:bbox` | `proj:bbox` is clipped to the **data** and, for the 10–29° rotated sortie grids, circumscribes and overstates the occupied region | `area_recompute.py`, `stage3_drcog_void.py` |
| `462.2555 ft` is the median `proj:bbox` **width**, not a footprint side | using it as "one full cell" overstates the release area by 1.3% | `area_recompute.py` foil table |
| interior tiles only for sampling | edge tiles fill only part of their footprint, corrupting the ground surface and every rate denominator | `interior_ids` |
| DRCOG column **45 is east of 46** (`gy` decreases eastward) | the gap sign inverts and the void reads as an overlap | `stage3_drcog_void.py`, asserted at runtime |
| memory discipline in the download loop | an earlier 16-worker run was OOM-killed by holding `laspy` objects across threads | `fetch_xyz` |

### Footprint size: use the polygon, never its bounding box

The tile footprint is a rectangle that is very slightly rotated relative to the projected
axes (0.36 deg in DRCOG). Its *axis-aligned bounding box* is therefore wider than the
rectangle itself, and the two are easy to confuse:

* footprint edge, measured on the polygon: **139.9615 m** (139.93-139.97 m across sorties)
* grid pitch, from footprint centres of adjacent columns: **139.9575 m**
* axis-aligned bounding box of the same polygon: **140.84 m**, because
  139.9615 x (cos 0.36 + sin 0.36) = 140.838

Consequences. Adjacent footprints **abut**, agreeing at their shared edge to about 4 mm;
they do not overlap by ~0.9 m, which is what the bounding-box figure would imply. And
`82,429 x (140.84 m)^2 = 1634.96 km^2` is *not* the release area: the published figure,
1610.50 km^2, is the sum of the polygon areas, and it is what yields the published
87.00 points/m^2. `area_recompute.py` prints the bounding-box and count-times-cell
variants deliberately, as foils, so the difference stays visible.

Two independent checks confirm 139.96 m rather than 140.84 m: the polygon-area sum
reproduces 1610.50 km^2 and 87.00 points/m^2 exactly, and the occupied-to-footprint ratio
for unaffected DRCOG columns comes out at 1.013 with a ~139.96 m denominator, against
~1.001 with 140.84 m.

Use polygon edges, or polygon areas. Never the bounding box.

### Methods that do **not** work — do not reintroduce them

1. **Footprint polygons cannot show the void.** They are nominal tiling geometry; the
   polygons of columns 45 and 46 abut, so their union implies continuous coverage across
   a band that holds no data. `stage3_drcog_void.py --mode bbox` prints this failure
   explicitly (`FAILED METHOD 1`) so it stays falsified rather than forgotten.
2. **Per-tile occupancy indexed from the tile's own minimum occupied easting cannot show
   it.** With the array origin at the first occupied cell, a region with no data lies
   outside the array and can never register as empty. This produced a false "density
   deficit, not a void" conclusion twice.
3. **Any test requiring a fully empty row or column returns zero**, because the void
   edges run diagonally across tiles (it is a flight-line boundary). Occupancy must be
   assessed in 2D over the *nominal* extent, or via the `proj:bbox` comparison used here.

---

## Outputs

Scripts print a human-readable report to stdout and take an optional `--out` for a
machine-readable copy (JSON or CSV depending on the script). `stage2_noise_sample.py`
writes JSONL, one record per tile, appended and fsynced as each tile completes.

```
validation/out/
  stage2_tiles.jsonl              per-tile noise metrics (full run)
  stage2_tiles_n5.jsonl           per-tile noise metrics (smoke run)
  checks_*.jsonl                  per-tile metrics from the validation checks
  void_bbox.json  void_headers.json
```
