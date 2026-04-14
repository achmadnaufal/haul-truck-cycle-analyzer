# Haul Truck Cycle Analyzer

Haul truck cycle time analysis, bottleneck identification, and fleet-level
productivity optimization for open-pit mining operations.

## Features

- Data ingestion from CSV and Excel input files
- Automated KPI calculation: cycle time, productivity (t/h), utilization
- Bottleneck identification per cycle (loading, hauling, dumping, return, queue)
- Fleet-level summary statistics per truck
- Edge-case handling: zero times, negative distances, single-truck fleets
- Sample dataset for quick exploration
- Full pytest test suite (80%+ coverage target)

## Installation

```bash
pip install -r requirements.txt
```

For running tests, also install pytest:

```bash
pip install pytest
```

## Quick Start

```python
from src.main import HaulTruckAnalyzer

analyzer = HaulTruckAnalyzer()

# Load and analyse the bundled sample dataset
result = analyzer.run("demo/sample_data.csv")

print(f"Cycles analysed : {result['total_records']}")
print(f"Mean productivity: {result['means']['productivity_tph']:.1f} t/h")
print(f"Bottleneck breakdown: {result['bottleneck_distribution']}")
```

## Sample Data

A ready-to-use sample dataset lives at `demo/sample_data.csv` (20 rows).

Columns:

| Column | Description |
|---|---|
| `cycle_id` | Unique identifier for each cycle |
| `truck_id` | Truck identifier |
| `truck_model` | Model name (e.g. CAT 793F) |
| `load_tonnes` | Payload per cycle (metric tonnes) |
| `loading_time_min` | Time spent loading at the shovel (minutes) |
| `hauling_time_min` | Time hauling loaded to the dump (minutes) |
| `dumping_time_min` | Time at the dump point (minutes) |
| `return_time_min` | Empty return travel time (minutes) |
| `queue_time_min` | Time waiting in queue (minutes) |
| `total_cycle_min` | Total cycle duration (minutes) |
| `distance_km` | One-way haul distance (km) |
| `route` | Route identifier |
| `shift` | Day or Night shift |
| `date` | Cycle date (YYYY-MM-DD) |

### Example Analysis

```python
import pandas as pd
from src.main import HaulTruckAnalyzer, fleet_summary

analyzer = HaulTruckAnalyzer()
df = analyzer.load_data("demo/sample_data.csv")

# Preprocess and enrich with computed KPIs
preprocessed = analyzer.preprocess(df)
enriched = analyzer.enrich(preprocessed)

# Per-truck fleet summary
summary = fleet_summary(enriched)
print(summary[["cycle_count", "avg_cycle_min", "total_tonnes", "avg_productivity_tph"]])

# Bottleneck breakdown across all cycles
print(enriched["bottleneck"].value_counts())
```

Expected output (approximate):

```
        cycle_count  avg_cycle_min  total_tonnes  avg_productivity_tph
truck_id
TRK01             5         50.300        1111.5               264.4
TRK02             4         56.950         751.0               196.7
TRK03             4         53.525         868.5               243.3
TRK04             4         65.375        1158.0               267.0
TRK05             4         48.650         845.5               261.2

bottleneck
hauling_time_min    20
dtype: int64
```

## Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_analyzer.py::TestComputeCycleTime::test_normal_row_sums_all_time_columns PASSED
tests/test_analyzer.py::TestComputeCycleTime::test_zero_times_returns_zero PASSED
...
================================ 25 passed in 0.42s ================================
```

## Project Structure

```
haul-truck-cycle-analyzer/
├── src/
│   ├── __init__.py
│   ├── main.py            # Core analysis logic and KPI helpers
│   └── data_generator.py  # Synthetic data generator
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py   # pytest unit and integration tests
├── demo/
│   └── sample_data.csv    # Ready-to-use 20-row sample dataset
├── examples/
│   └── basic_usage.py     # End-to-end usage example
├── data/                  # Drop real data files here (gitignored)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## License

MIT License — free to use, modify, and distribute.
