# INSIGHTS quality-control and validation analyses

Reference implementations of the analyses reported in the *Technical Validation* section
of the INSIGHTS data descriptor. Each script runs against the public release, so you can
reproduce a published figure, check a result for yourself, or apply the same measurement
to whichever part of the dataset you plan to use.

INSIGHTS-LiDAR is a research-grade product. It carries residual noise that differs
substantially between sorties, one known coverage gap, and vertical offsets between
sorties. None of this is flagged in the point files themselves — `Classification` is zero
for every point — so if it matters to your work, you have to measure it. These scripts are
how the published measurements were made.

## What you might use this for

| if you want to | start with |
|---|---|
| decide whether a sortie is clean enough for your analysis | the per-sortie table from `noise/stage2_report.py`, or the noise table in the paper |
| detect and remove noise in the tiles you are using | the method in `noise/stage2_noise_sample.py`, described under [Detecting noise](#detecting-noise) |
| compute area or point density correctly | `coverage/area_recompute.py`, and [Footprints, bounding boxes, and area](#footprints-bounding-boxes-and-area) |
| find where data are missing rather than merely sparse | `coverage/stage3_drcog_void.py` |
| combine or difference two sorties | `georegistration/characterize_offsets.py`, and [Combining sorties](#combining-sorties) |
| confirm a published number before relying on it | [Reproducing the published results](#reproducing-the-published-results) |

---

## Setup

The scripts need only the tile index and public S3 — no credentials, and no PDAL. The
project's `pyproject.toml` lists `pdal`, whose Python bindings require a local PDAL C++
install, so the simplest route is a small standalone environment:

```sh
cd <repo root>
uv venv .venv-validation --python 3.11
uv pip install --python .venv-validation/bin/python \
    pandas pyarrow numpy laspy lazrs shapely boto3 pyproj scipy
```

`lazrs` lets `laspy` read COPC LAZ. `scipy` provides rank correlation. No script produces
figures, so `matplotlib` is not needed.

Examples below use `PY=.venv-validation/bin/python` from the repo root. Every script
accepts `--help`, and `--index` if your copy of `items.parquet` lives elsewhere.

---

## The scripts

| script | needs network | what it measures |
|---|---|---|
| `noise/stage1_index_screen.py` | no | per-sortie elevation excursion and terrain relief, from the index alone |
| `noise/stage2_noise_sample.py` | yes | per-tile noise metrics over a seeded sample of interior tiles |
| `noise/stage2_report.py` | no | aggregates those metrics into the per-sortie noise table |
| `noise/stage2_checks.py` | yes | the negative control and four sensitivity checks |
| `coverage/stage3_drcog_void.py` | `--mode headers` only | the DRCOG coverage gap, and that `proj:bbox` describes the data |
| `coverage/area_recompute.py` | no | release-wide footprint area and mean point density |
| `georegistration/characterize_offsets.py` | yes | vertical offsets between overlapping sorties |
| `georegistration/test_relative_accuracy.py` | yes | relative versus absolute accuracy over co-observed ground |
| `georegistration/verify_offset.py` | yes | whether an offset is a constant shift rather than a warp |
| `georegistration/summarize_offsets.py` | no | aggregates measured offsets by sortie pair and by sortie |
| `georegistration/inspection_pairs.py` | no | picks example tile pairs to inspect visually, with COPC links |

Three need no downloads and finish in seconds, working entirely from the published
GeoParquet index: `stage1_index_screen.py`, `area_recompute.py`, and
`stage3_drcog_void.py --mode bbox`. They are the cheapest way to confirm your copy of the
index behaves as the paper describes.

---

## Methodology

### Detecting noise

Two kinds of spurious return appear in the release, and they need different treatment.

**Sparse noise** is what ASPRS classes 7 and 18 describe: isolated returns above or below
the terrain. Detecting it takes two conditions together.

1. *Height above ground.* A coarse terrain reference is estimated per tile as the 5th
   percentile of elevation within 5 m cells, hole-filled from neighbours, then subtracted.
   Raw elevation will not serve: within-tile elevation range is dominated by real terrain
   slope, which in the mountain corridors reaches hundreds of feet across a single 140 m
   tile. A low percentile rather than the minimum keeps sparse below-surface noise from
   dragging the reference down beneath the noise itself.
2. *Isolation.* A point counts as isolated when its 3×3×3 neighbourhood of 2 m voxels
   holds fewer than 20 points. Because this makes no reference to height, tall real
   structures survive it.

High noise is then height above ground above 20 m *and* isolated; low noise is below −2 m
*and* isolated, that threshold chosen to exceed the 0.36 m vertical registration RMSE.

**Structured artifact layers** — the banded surfaces well above the terrain in the I70 and
I80 sorties — are a different problem, because they are locally dense and so no sparsity
test finds them. In I70E, 9–16% of points lie above 20 m above ground while only 1–2% of
those register as isolated; loosening the isolation threshold twentyfold still captures
only about a third of them. What separates such a layer from a real surface is *areal*
density in plan view: a roof or tree canopy is sampled at close to the tile's full
density, whereas an artifact layer deposits only a few points per square metre over the
ground it covers. The scripts therefore report extent and areal density for elevated
returns **without** the isolation filter, alongside the isolation-filtered rates.

If you are filtering your own data, the practical consequence is that a conventional
statistical outlier filter will remove most of the sparse noise and barely touch the
banded layers. Those need a height-above-ground criterion combined with an areal-density
threshold. In the mountain corridors that is comparatively safe, since no real structure
approaches the 137–191 m heights where the layers sit. Over the Denver or Salt Lake urban
cores the same filter would delete genuine buildings, which reach 154 m above ground in
the tiles examined here — height thresholds do not transfer between sorties.

Sampling uses **interior tiles only**, meaning tiles whose eight grid neighbours are all
present in the same sortie. An edge tile may fill only part of its footprint, which would
corrupt both the terrain reference and every rate denominator. Samples are drawn with a
per-sortie seed from a stable hash, so a given `--n` and `--seed` always select the same
tiles; `--check-sample` verifies that across subprocesses, and `--print-sample` lists the
chosen ids without downloading anything.

### Footprints, bounding boxes, and area

This matters to anyone computing area, density, or coverage.

A tile footprint is a rectangle slightly rotated relative to the projected axes — 0.36° in
DRCOG. Its axis-aligned bounding box is therefore wider than the rectangle itself:

* footprint edge, measured on the polygon: **139.96 m** (139.93–139.97 m across sorties)
* grid pitch between adjacent footprint centres: **139.96 m**
* axis-aligned bounding box of that same polygon: **140.84 m**, since
  139.96 × (cos 0.36° + sin 0.36°) = 140.84

Adjacent footprints therefore **abut**, agreeing at their shared edge to a few
millimetres. And `82,429 × (140.84 m)² = 1634.96 km²` is *not* the release area: the
published figure, **1610.50 km²**, is the sum of the footprint polygon areas, and it is
what yields the published **87.00 points/m²**. `area_recompute.py` prints the bounding-box
and count-times-cell variants alongside the correct figure so the difference stays visible.

Two independent checks confirm the polygon edge over the bounding box: the polygon-area
sum reproduces 1610.50 km² and 87.00 points/m² exactly, and the occupied-to-footprint
ratio for unaffected DRCOG columns comes out at 1.013 with a 139.96 m denominator against
1.001 with 140.84 m.

The footprint polygon and `proj:bbox` are also different things. `proj:bbox` is clipped to
the returns actually present, which makes it a useful occupied-extent estimate — but it is
axis-aligned, and several sortie grids are rotated by 10–29°, so for those it circumscribes
and overstates the occupied region. Summed across the release it gives 1790.10 km² against
the footprints' 1610.50 km². Treat it as an upper bound except in the near-aligned sorties.

### Finding missing data

A footprint polygon says where a tile sits in the tiling scheme, not where the returns
are. In DRCOG, two adjacent tile columns are separated by a band roughly 110 m wide
containing no data, even though their footprints abut and so imply continuous coverage.

`stage3_drcog_void.py` locates it by comparing `proj:bbox` between adjacent grid columns
row by row, which is valid because DRCOG's grid is aligned to within 0.4°. `--mode headers`
corroborates the result cheaply across the whole sortie by reading only the 375-byte LAS
public header block of each file over an HTTP range request, confirming that the header
bounding box matches `proj:bbox` exactly for all 36,221 tiles.

Three approaches that do **not** find this gap, worth knowing if you write your own check:

1. Footprint polygons cannot show it — they are nominal geometry and abut across the band.
   `--mode bbox` reports this explicitly so the point is not lost.
2. Per-tile occupancy indexed from a tile's own first occupied cell cannot show it: a
   region with no data falls outside the array and never registers as empty.
3. Any test requiring a fully empty row or column finds nothing, because the gap edges run
   diagonally across tiles — it is a flight-line boundary, not a tile boundary. Occupancy
   has to be assessed in two dimensions over the nominal extent.

### Combining sorties

Sorties overlap, so the same ground is sometimes observed twice and the two independent
looks can be differenced directly. The `georegistration/` scripts do this over flat, hard,
fully sampled ground: a 2 m grid across the overlap, a per-cell ground proxy at the 10th
percentile of elevation, and a cell used only where both sorties have at least 30 points
and a within-cell spread under 0.5 m.

The distinction they draw is between agreement in *shape* and agreement in *height*.
Within a single overlap the two surfaces are very nearly parallel, agreeing to a few
centimetres — far better than the absolute georegistration RMSE, as expected of relative
geometry. Between sorties, however, they are offset vertically by metre order.

So if you mosaic sorties, difference them for change detection, or build an elevation
model across a sortie boundary, estimate and remove a vertical offset first rather than
merging the products as published.

Two companions work from the measurement file that `characterize_offsets.py` writes, so they
need no network access and are cheap to re-run. `summarize_offsets.py` aggregates the offsets
by sortie pair and by sortie. `inspection_pairs.py` is a verification aid rather than an
analysis: for each sortie pair it selects a typical and an extreme example tile pair and prints
COPC links plus the overlap centroid, so the offsets can be confirmed by eye instead of taken
on trust. Its selection rule is in the code — closest to the median, and largest deviation from
it — so the choice of examples is auditable rather than hand-picked. Both default to
`out/offsets.jsonl`, which is a run artifact and not tracked in git, so run
`characterize_offsets.py` first or point `--pairs` at your own measurement file.

What the measurements show, beyond the offsets themselves: fitting a plane to the difference
surface returns a nonzero gradient in every pair, a median of 0.7 mm/m and up to 13 mm/m. The
sorties are not merely offset in elevation but slightly askew, which is why an offset can vary
by metres across one sortie pair and why no single per-sortie number describes the relationship.

`characterize_offsets.py` measures every overlapping sortie pair, then asks whether the
offsets are a property of each sortie rather than of each pair. Overlaps form a graph on
sorties; within each connected component the script solves `d_ij = b_i − b_j` for a single
bias per sortie, gauged so the biases sum to zero. Where a component contains more
overlapping pairs than sorties minus one, the system is over-determined and the residuals
test that question directly: small residuals mean a per-sortie vertical correction exists,
large ones mean the offsets are specific to a pair or a location. The script excludes the
western third of `I80P2`, where dense noise would corrupt a ground estimate.

---

## Reproducing the published results

### Free and exact

These read the index only and reproduce published figures precisely.

```sh
PY=.venv-validation/bin/python

$PY validation/coverage/area_recompute.py          # 1610.50 km^2, 87.00 points/m^2
$PY validation/coverage/stage3_drcog_void.py --mode bbox
$PY validation/noise/stage1_index_screen.py
```

`area_recompute.py` reports 1610.50 km² by footprint-polygon sum, agreeing to six decimal
places across an equal-area projection, the native State Plane CRS, and geodesic
integration, together with the 140,116,442,116-point total that gives 87.00 points/m².

`stage3_drcog_void.py --mode bbox` finds the coverage gap in all 246 grid rows where both
columns exist, median width 110.7 m, centred at 105°00'34.3"W and running the full ~34 km
extent of the sortie. It also shows the gap is missing *area* rather than reduced density:
the affected column holds 36.7% of the normal point count over 37.1% of the footprint
area, so density within the occupied part matches the rest of the sortie to 0.2%.

### The header cross-check

Roughly 375 bytes per tile, about ten minutes for all of DRCOG. Add `--limit 500` for a
quick pass.

```sh
$PY validation/coverage/stage3_drcog_void.py --mode headers --workers 32
```

### The negative control

Four downloads, and the most informative single check in the noise method: it is what
shows real buildings are not counted as noise.

```sh
$PY validation/noise/stage2_checks.py negative-control
```

On the four downtown Denver tiles with the largest elevation excursion in DRCOG, 10–20% of
points lie above 20 m above ground and reach 154 m, yet only 0.03–0.8% of those elevated
returns are flagged as isolated, and their areal-density ratio is 0.60–1.50, close to full
sampling. Artifact layers score 0.004–0.158 on the same ratio. The two populations do not
overlap, the nearest values differing by about a factor of four.

### The per-sortie noise table

510 tiles at roughly 12 MB each, so expect a few gigabytes and up to an hour depending on
throughput. Measurements are checkpointed to JSONL per tile, and an interrupted run
resumes.

```sh
$PY validation/noise/stage2_noise_sample.py --n 30 --workers 16 --check-sample \
    --out validation/out/stage2_tiles.jsonl
$PY validation/noise/stage2_report.py --tiles validation/out/stage2_tiles.jsonl
```

A smaller `--n` runs much faster and is fine for checking the pipeline works, but it will
not reproduce the published table; `stage2_report.py` says so when `n < 30`.

### Sensitivity checks

```sh
$PY validation/noise/stage2_checks.py iso-sweep       # isolation threshold
$PY validation/noise/stage2_checks.py ground-sweep    # terrain percentile
$PY validation/noise/stage2_checks.py i80p2-thirds    # west-to-east heterogeneity
$PY validation/noise/stage2_checks.py drcog-water     # 50 downloads
$PY validation/noise/stage2_checks.py all
```

The ground-percentile sweep is worth running if you adapt the terrain reference: published
rates move by under 1% as the percentile varies from the 1st to the 50th.

### Inter-sortie offsets

```sh
$PY validation/georegistration/characterize_offsets.py --per-combo 10 --workers 10 \
    --out validation/out/offsets.jsonl
```

Two downloads per tile pair. `--per-combo` sets how many pairs are measured per sortie
combination; pairs are spread along each overlap corridor so an estimate is not drawn from
a single location.

---

## How to read the results

Some of these analyses support a claim and some deliberately refute one. Both are useful,
and the difference matters if you cite them.

**Established.** The per-sortie noise differences, spanning three orders of magnitude,
with I70E worst on every measure. The areal-density discriminator, checked in both
directions against real buildings and against artifact layers. The DRCOG coverage gap. The
release-wide area and density figures. The exact correspondence between LAS headers and
`proj:bbox`.

**Not established, and the paper claims neither.**

*Terrain relief does not predict noise.* Elevation excursion measured from the index does
correlate with local relief, but the strongest correlation in the release belongs to
FrontRange (ρ = 0.747), a clean sortie. The metric tracks real terrain slope, so a large
excursion is not evidence of an artifact. This is why noise is measured in height above
ground on the points themselves rather than from index statistics.

*The DRCOG water association is unconfirmed.* Comparing DRCOG interior tiles from the
lowest point-count decile against density-matched controls gives the same proportion of
noise-bearing tiles in both. The observation that DRCOG's sparse high noise sits over
water comes from visual inspection, and the paper reports it as such. Masking water
remains sensible on independent grounds, since water returns few and unreliable
detections.

---

## Implementation notes

Details that are easy to get wrong, and that explain why the code looks as it does. They
matter mainly if you adapt these scripts.

| detail | why |
|---|---|
| `proj:bbox` and `bbox` are JSON **strings** in the index | parse them; iterating one as a list yields characters |
| footprint `geometry` is WKB with a **6-coordinate XYZ ring** | 4 corners, a closing vertex, and a duplicate of it |
| projected CRSs are **US survey feet** | 0.3048006096012192 m/ft, not the international foot |
| sampling seeds use `crc32`, not `hash()` | `PYTHONHASHSEED` randomizes `str` hashing per process, so `hash()` redraws the sample on every run |
| the plan grid takes **no `+1`** | a trailing row or column no point can occupy inflates empty-cell fractions and manufactures voids |
| areal-density metrics **exclude** the isolation filter | the artifact layer is locally dense, so filtering by isolation first discards the population being measured |
| vertical band statistics **include only** isolated points | otherwise vegetation dominates the histogram and fakes a treetop-height band |
| DRCOG grid column **45 lies east of 46** (`gy` decreases eastward) | reversed, the gap changes sign and reads as an overlap |
| coordinate arrays are copied and the `laspy` object released at once | holding point records across worker threads exhausts memory on a modest machine |

Throughput is network-bound rather than CPU-bound — about 12 MB per tile against roughly
0.3 s of computation — so 16–24 workers is usually the useful range, and more brings
little.

---

## Outputs

Each script prints a readable report to stdout and takes an optional `--out` for a
machine-readable copy. `stage2_noise_sample.py` writes JSONL, one record per tile, flushed
as each tile completes so that a long run can resume.

```
validation/out/
  stage2_tiles.jsonl        per-tile noise metrics
  checks_*.jsonl            per-tile metrics from the validation checks
  offsets.jsonl             per-pair inter-sortie vertical offsets
  void_bbox.json            coverage-gap geometry
  void_headers.json         LAS header cross-check summary
```

Run artifacts are not tracked in git; regenerate them with the commands above. The figures
as published appear in the data descriptor.
