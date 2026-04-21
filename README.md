# ARPA-I INSIGHTS Dataset

A large-scale, high-density (~85 pts/m²) geiger-mode aerial LiDAR dataset covering over 1,600 km² across the Salt Lake City, UT and Denver, CO metropolitan regions.

## Overview

- **Coverage:** 1,600+ km² across Salt Lake City and Denver
- **Point Count:** ~140 billion points in 82,429 LAS tiles
- **Resolution:** ~30 cm vertical/horizontal spacing
- **Collection Date:** June 2025
- **Format:** LAS/LAZ point clouds with STAC catalog (GeoParquet index)
- **License:** CC-BY-4.0 (data), MIT (code)

## Dataset Structure

```
s3://arpa-i-insights/
├── lidar/
│   ├── data/              # L3 LAS files (one folder per sortie)
│   │   ├── DRCOG/
│   │   ├── SLC/
│   │   ├── I15South/
│   │   ├── I25N/
│   │   ├── I70*/         # Multiple segments
│   │   ├── I80*/         # Multiple segments
│   │   └── FrontRange/
│   └── stac/
│       ├── index/items.parquet   # STAC GeoParquet index
│       ├── items/               # STAC items (JSON per sortie)
│       └── collection.json      # Full STAC collection
```

## Key Features

- Tiled LAS storage for cloud-optimized access
- STAC catalog with GeoParquet index for fast spatial queries
- Two coordinate systems: NAD83(2011) / Colorado Central (EPSG:6430) and NAD83(2011) / Utah Central (EPSG:6626)

## Quick Start

```python
import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

bucket = "arpa-i-insights"
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Load STAC catalog
stac_df = pd.read_parquet("s3://arpa-i-insights/lidar/stac/index/items.parquet")
print(f"Available tiles: {len(stac_df):,}")
```

## Usage Examples

- **Notebook:** See [`examples/get-to-know-ARPA-I-INSIGHTS.ipynb`](examples/get-to-know-ARPA-I-INSIGHTS.ipynb) for a guided tour
- **Analysis Script:** See [`examples/scripts/analyze_las.py`](examples/scripts/analyze_las.py) for ground classification and DEM generation

## Tools & Libraries

- **Point Cloud:** laspy, PDAL, CloudCompare
- **Spatial Data:** geopandas, pyarrow
- **Cloud Access:** boto3

## Sponsors & Maintainers

- **Sponsor:** U.S. Department of Transportation ARPA-I
- **Maintainer:** MIT Lincoln Laboratory

## Citation

```
ARPA-I INSIGHTS LiDAR Dataset (2026). MIT Lincoln Laboratory.
U.S. Department of Transportation ARPA-I.
Available at: https://registry.opendata.aws/arpa-i-insights/
```
