# Haul Truck Cycle Analyzer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-orange)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)

Haul truck cycle time analysis, bottleneck identification, and fleet-level
productivity optimization for open-pit mining operations.

Supports both a **standard schema** (pre-split time components) and a
**timestamp schema** (ISO-8601 start/end columns with payload and distance
aliases), covering common export formats from fleet management systems used
with CAT 777, CAT 793F, Komatsu HD785, Liebherr T 284, and similar trucks.

---

## Features

- Data ingestion from CSV and Excel input files
- Automated KPI calculation: cycle time, productivity (t/h), utilization
- Bottleneck identification per cycle (loading, hauling, dumping, return, queue)
- Fleet-level summary statistics per truck
- Comprehensive input validation: missing truck IDs, out-of-order timestamps,
  negative payloads, negative cycle times, duplicate trip records
- Column alias normalisation (`load_time_min` -> `loading_time_min`,
  `payload_tonnes` -> `load_tonnes`, `haul_distance_km` -> `distance_km`)
- Immutable data-flow design: no in-place mutation of input DataFrames
- Edge-case handling: zero times, negative distances, single-truck fleets
- 20-row realistic sample dataset (open-pit coal mining, multi-truck fleet)
- Full pytest test suite with 80 %+ coverage target

---

## Quick Start

### Installation

```bash
git clone https://github.com/achmadnaufal/haul-truck-cycle-analyzer.git
cd haul-truck-cycle-analyzer
pip install -r requirements.txt
```

### Run the bundled sample

```python
from src.main import HaulTruckAnalyzer

analyzer = HaulTruckAnalyzer()

# Load, validate, and analyse the bundled 20-row sample dataset
result = analyzer.run("demo/sample_data.csv")

print(f"Cycles analysed    : {result['total_records']}")
print(f"Mean productivity  : {result['means']['productivity_tph']:.1f} t/h")
print(f"Bottleneck breakdown: {result['bottleneck_distribution']}")
```

### Step-by-step pipeline

```python
from src.main import HaulTruckAnalyzer, fleet_summary

analyzer = HaulTruckAnalyzer()

# 1. Load raw data
df = analyzer.load_data("demo/sample_data.csv")

# 2. Validate before processing (raises ValueError on bad data)
analyzer.validate(df)

# 3. Preprocess: normalise column names, clamp negatives
preprocessed = analyzer.preprocess(df)

# 4. Enrich: compute cycle time, productivity, bottleneck
enriched = analyzer.enrich(preprocessed)

# 5. Per-truck fleet summary
summary = fleet_summary(enriched)
print(summary[["cycle_count", "avg_cycle_min", "total_tonnes", "avg_productivity_tph"]])

# 6. Bottleneck distribution across all cycles
print(enriched["bottleneck"].value_counts())
```

### Using the timestamp schema

The analyzer also accepts data exports with `timestamp_start` / `timestamp_end`
columns and common alias column names:

```python
import pandas as pd
from src.main import HaulTruckAnalyzer

df = pd.read_csv("demo/sample_data.csv")
# Columns: truck_id, trip_id, timestamp_start, timestamp_end,
#          load_time_min, haul_time_min, dump_time_min, return_time_min,
#          payload_tonnes, haul_distance_km, material_type, pit_name

analyzer = HaulTruckAnalyzer()
result = analyzer.run("demo/sample_data.csv")
print(result["fleet_summary"])
```

### Configuration options

```python
analyzer = HaulTruckAnalyzer(config={
    "shift_duration_min": 720,     # 12-hour shift (default)
    "strict_validation": True,     # raise on negative times (default)
})
```

---

## Sample Output

Running `analyzer.run("demo/sample_data.csv")` on the bundled dataset produces
output similar to:

```
Cycles analysed    : 20
Mean productivity  : 252.3 t/h
Bottleneck breakdown: {'hauling_time_min': 20}
```

Fleet summary table:

```
          cycle_count  avg_cycle_min  total_tonnes  avg_productivity_tph
truck_id
TRK01               5         50.30        1111.5                 264.4
TRK02               4         56.95         751.0                 196.7
TRK03               4         53.53         858.5                 243.3
TRK04               4         65.38        1158.0                 267.0
TRK05               4         48.65         845.5                 261.2
```

Bottleneck distribution (all cycles):

```
bottleneck
hauling_time_min    20
dtype: int64
```

---

## Sample Data

A ready-to-use sample dataset lives at `demo/sample_data.csv` (20 rows,
representing three shifts across five trucks at an open-pit coal mine).

| Column | Description |
|---|---|
| `truck_id` | Truck identifier (e.g. TRK01) |
| `trip_id` | Unique trip identifier per cycle |
| `timestamp_start` | ISO-8601 cycle start datetime |
| `timestamp_end` | ISO-8601 cycle end datetime |
| `load_time_min` | Time spent loading at the shovel (minutes) |
| `haul_time_min` | Loaded haul travel time (minutes) |
| `dump_time_min` | Time at the dump/crusher point (minutes) |
| `return_time_min` | Empty return travel time (minutes) |
| `payload_tonnes` | Payload per cycle (metric tonnes) |
| `haul_distance_km` | One-way haul distance (km) |
| `material_type` | Material moved (Coal or Overburden) |
| `pit_name` | Source pit identifier |
| `queue_time_min` | Time waiting in shovel queue (minutes) |
| `total_cycle_min` | Total cycle duration (minutes) |
| `shift` | Day or Night shift |
| `date` | Cycle date (YYYY-MM-DD) |

Truck models represented:

| Truck ID | Model | Nominal Payload |
|---|---|---|
| TRK01 | CAT 793F | ~227 t |
| TRK02 | Komatsu 830E | ~186 t |
| TRK03 | CAT 793F | ~227 t |
| TRK04 | Liebherr T 284 | ~290 t |
| TRK05 | CAT 793F | ~227 t |

---

## New: Fleet Match Factor Calculator

The match factor (MF) quantifies whether the loader and truck fleet are balanced.
An MF of 1.0 is ideal; MF < 0.90 means the loader is the bottleneck
(trucks waiting); MF > 1.10 means trucks are the bottleneck (loader sits idle).

### Step-by-step usage

```python
import pandas as pd
from src.fleet_match_factor_calculator import calculate_fleet_match_factor

# 1. Load enriched cycle data
df = pd.read_csv("demo/sample_data.csv")

# 2. Compute match factor per pit
#    loader_cycle_time_min: loader swing-and-load time per pass
#    n_passes: passes required to fill one truck (1 for large rope shovels)
report = calculate_fleet_match_factor(
    df,
    loader_cycle_time_min=12.0,
    n_passes=1,
    pit_col="pit_name",       # column identifying each loading zone
    truck_col="truck_id",     # column identifying trucks
)

# 3. Inspect fleet-wide summary
print(f"Overall match factor : {report.overall_match_factor:.3f}")
print(f"Under-trucked pits   : {report.n_under_trucked}")
print(f"Over-trucked pits    : {report.n_over_trucked}")
print(f"Balanced pits        : {report.n_balanced}")

# 4. Per-pit detail
for result in report.pit_results:
    print(
        f"{result.pit_name}: {result.n_trucks} trucks, "
        f"MF={result.match_factor:.3f}, condition={result.condition}"
    )

# 5. Export as DataFrame for downstream analysis
df_report = report.to_dataframe()
print(df_report.to_string(index=False))
```

### Standalone match factor for a single pit

```python
from src.fleet_match_factor_calculator import compute_match_factor

mf = compute_match_factor(
    n_trucks=4,
    truck_cycle_time_min=50.0,
    loader_cycle_time_min=12.0,
    n_passes=1,
)
print(f"Match factor: {mf}")   # → 0.96 (balanced)
```

---

## New: Queue Time Analyzer

The queue time analyzer quantifies what fraction of each truck's cycle is
spent waiting in the loader queue, ranks the worst offenders, and tags each
truck with a severity bucket. Excessive queueing is a leading indicator of
over-trucking, poor shovel availability, or unbalanced dispatch.

Severity buckets (queue minutes / cycle minutes):

| Bucket    | Range          | Operational meaning                     |
|-----------|----------------|------------------------------------------|
| low       | <= 5 %         | Healthy fleet balance                    |
| moderate  | 5 % - 15 %     | Normal congestion, monitor               |
| high      | 15 % - 25 %    | Investigate dispatch and loader uptime   |
| critical  | > 25 %         | Take immediate action (re-route trucks)  |

### Step-by-step usage

```python
import pandas as pd
from src.queue_time_analyzer import analyze_queue_time

# 1. Load enriched cycle data
df = pd.read_csv("demo/sample_data.csv")

# 2. Run the analyzer (uses total_cycle_min by default)
report = analyze_queue_time(df)

# 3. Inspect fleet-wide queueing
print(f"Fleet queue ratio   : {report.fleet_queue_ratio:.3f}")
print(f"Worst-offender truck: {report.worst_truck}")
print(f"Critical trucks     : {report.n_critical_trucks}")

# 4. Per-truck detail (sorted worst first)
for stat in report.truck_stats:
    print(
        f"{stat.truck_id}: {stat.n_cycles} cycles, "
        f"avg queue {stat.avg_queue_min:.1f} min, "
        f"ratio={stat.queue_ratio:.3f} ({stat.severity})"
    )

# 5. Export as DataFrame for dashboards
print(report.to_dataframe().to_string(index=False))
```

### Standalone severity classification

```python
from src.queue_time_analyzer import classify_queue_severity

classify_queue_severity(0.03)   # 'low'
classify_queue_severity(0.18)   # 'high'
classify_queue_severity(0.40)   # 'critical'
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_analyzer.py::TestComputeCycleTime::test_normal_row_sums_all_time_columns PASSED
tests/test_analyzer.py::TestComputeCycleTime::test_zero_times_returns_zero PASSED
tests/test_analyzer.py::TestComputeCycleTime::test_negative_values_clamped_to_zero PASSED
tests/test_analyzer.py::TestComputeProductivity::test_standard_productivity PASSED
...
tests/test_analyzer.py::TestTimestampSchema::test_full_pipeline_on_sample_csv PASSED
================================ 50+ passed in 0.60s ================================
```

Run with coverage:

```bash
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

---

## Project Structure

```
haul-truck-cycle-analyzer/
├── src/
│   ├── __init__.py
│   ├── main.py            # Core analysis logic, KPI helpers, validation
│   └── data_generator.py  # Synthetic data generator (standard schema)
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py   # pytest unit + integration tests (50+ cases)
├── demo/
│   └── sample_data.csv    # 20-row realistic open-pit coal mining dataset
├── examples/
│   └── basic_usage.py     # End-to-end usage example
├── data/                  # Drop real data files here (gitignored)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

### Key modules

`src/main.py` exposes:

| Symbol | Type | Description |
|---|---|---|
| `HaulTruckAnalyzer` | class | Main analysis pipeline |
| `compute_cycle_time` | function | Sum time components for one row |
| `compute_productivity` | function | Tonnes-per-hour from payload + cycle |
| `identify_bottleneck` | function | Dominant time phase for one row |
| `compute_utilization` | function | Active / available shift time % |
| `fleet_summary` | function | Per-truck aggregation |

---

## License

MIT License — free to use, modify, and distribute.
