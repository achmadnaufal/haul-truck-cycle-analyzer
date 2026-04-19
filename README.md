# Haul Truck Cycle Analyzer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest%20206%20passed-orange)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)

Haul truck cycle time analysis, bottleneck identification, and fleet-level
productivity optimization for open-pit mining operations.

The analyzer ingests raw cycle records from fleet management systems (CAT
777E / 793F, Komatsu HD785 / 830E, Volvo A60H, Liebherr T 284, and similar),
decomposes each cycle into its constituent stages (load, haul, dump, return,
queue), and computes the KPIs that drive daily dispatch decisions: cycle
time, productivity (t/h), fleet match factor, queue-time severity, and the
fleet-wide bottleneck stage.

---

## Features

- **Data ingestion** from CSV and Excel, with alias normalisation across two
  common export schemas (standard and timestamp-based).
- **Automated KPI calculation**: cycle time, productivity (t/h), utilization,
  per-truck fleet summary.
- **Per-cycle bottleneck detection** (`src/main.py::identify_bottleneck`) --
  which stage dominated *this* cycle.
- **Fleet-wide cycle decomposition** (`src/cycle_decomposition.py`) -- mean,
  median, p95, min, max, std, and share-of-cycle for every stage, plus the
  dominant stage across the whole fleet.
- **Fleet match factor** (`src/fleet_match_factor_calculator.py`) --
  quantifies loader-to-truck balance; MF = 1 is ideal, MF < 0.9 = under-
  trucked, MF > 1.1 = over-trucked.
- **Queue time analyzer** (`src/queue_time_analyzer.py`) -- per-truck queue
  ratio and severity bucket (`low`, `moderate`, `high`, `critical`).
- **Outlier filter** (`src/outlier_filter.py`) -- IQR or z-score strategies
  for cleaning abnormal records before KPI computation.
- **Comprehensive input validation**: missing truck IDs, out-of-order
  timestamps, negative payload, negative cycle times, duplicate trip records.
- **Immutable data-flow design**: no in-place mutation of input DataFrames,
  frozen-dataclass result objects.
- **Edge-case handling**: zero cycles, negative durations, missing stage
  columns, NaN values, empty DataFrames, single-truck fleets.
- **20-row realistic sample dataset** bundled at `demo/sample_data.csv`.
- **pytest test suite** with 200+ cases covering unit, integration, and
  property-based edge cases.

---

## Installation

```bash
git clone https://github.com/achmadnaufal/haul-truck-cycle-analyzer.git
cd haul-truck-cycle-analyzer
pip install -r requirements.txt
```

Requires Python 3.9+, pandas >= 2.0, numpy >= 1.24.

---

## Quick Start

The bundled `demo/sample_data.csv` holds 20 realistic cycles across five
trucks (CAT 777E, Komatsu HD785, Volvo A60H) and three shovels.

### 1. Load, validate, run the full pipeline

```python
from src.main import HaulTruckAnalyzer

analyzer = HaulTruckAnalyzer()
result = analyzer.run("demo/sample_data.csv")

print(f"Cycles analysed    : {result['total_records']}")
print(f"Mean productivity  : {result['means']['productivity_tph']:.1f} t/h")
print(f"Bottleneck breakdown: {result['bottleneck_distribution']}")
```

### 2. Decompose the cycle and find the fleet bottleneck stage

```python
import pandas as pd
from src.main import HaulTruckAnalyzer
from src.cycle_decomposition import decompose_cycle

analyzer = HaulTruckAnalyzer()
raw = pd.read_csv("demo/sample_data.csv")
enriched = analyzer.enrich(analyzer.preprocess(raw))

report = decompose_cycle(enriched)
print(f"Mean cycle     : {report.mean_total_cycle_min:.1f} min")
print(f"Bottleneck     : {report.bottleneck_stage} "
      f"({report.bottleneck_share:.1%} of cycle)")
print(report.to_dataframe().to_string(index=False))
```

Example output on the bundled sample:

```
Mean cycle     : 32.3 min
Bottleneck     : haul (41.5% of cycle)
  stage          column  n_cycles  mean_min  median_min  p95_min  share_of_cycle
   load loading_time_min        20      3.33        3.35    4.205          0.1032
   haul hauling_time_min        20     13.43       13.65   16.810          0.4154
   dump dumping_time_min        20      1.95        2.00    2.555          0.0602
 return  return_time_min        20     10.51       10.50   13.330          0.3251
  queue   queue_time_min        20      3.11        2.90    5.270          0.0962
```

### 3. Check fleet match factor

```python
from src.cycle_decomposition import decompose_cycle
from src.fleet_match_factor_calculator import compute_match_factor

report = decompose_cycle(enriched)
mf = compute_match_factor(
    n_trucks=5,
    truck_cycle_time_min=report.mean_total_cycle_min,
    loader_cycle_time_min=6.5,   # effective loader swing+load per truck
    n_passes=1,
)
print(f"Match factor: {mf}")
# 1.0 is ideal; <0.90 under-trucked, >1.10 over-trucked.
```

---

## Fleet Match Factor

The match factor quantifies whether the loader and truck fleet are balanced.
An MF of 1.0 is ideal; MF < 0.90 means the loader is the bottleneck (trucks
waiting), MF > 1.10 means trucks are the bottleneck (loader sits idle).

```python
from src.fleet_match_factor_calculator import compute_match_factor

# 4 trucks, 50-min cycle, loader 12.5 min per pass, 1 pass -> MF exactly 1.0
mf = compute_match_factor(
    n_trucks=4,
    truck_cycle_time_min=50.0,
    loader_cycle_time_min=12.5,
    n_passes=1,
)
print(mf)  # 1.0
```

For a DataFrame-level view, use
`calculate_fleet_match_factor(df, loader_cycle_time_min, ...)` which groups
cycles by pit and returns an immutable `MatchFactorReport`.

---

## Cycle Decomposition (new)

The cycle decomposition module (`src/cycle_decomposition.py`) breaks each
cycle into its five canonical stages and returns rich per-stage statistics
plus the fleet-wide bottleneck:

- `mean_min`, `median_min`, `p95_min`, `min_min`, `max_min`, `std_min`
- `share_of_cycle` (fraction of average total cycle consumed)
- `bottleneck_stage` across the fleet (largest mean)
- `rank_stages_by_median(df)` -- alternative ranking robust to long tails

This complements the per-cycle `identify_bottleneck` helper: that answers
"which stage dominated *this* cycle?", while `decompose_cycle` answers
"which stage dominates the *fleet*?".

---

## Queue Time Analyzer

Severity buckets (queue minutes / cycle minutes):

| Bucket    | Range       | Operational meaning                       |
|-----------|-------------|--------------------------------------------|
| low       | <= 5 %      | Healthy fleet balance                      |
| moderate  | 5 % - 15 %  | Normal congestion, monitor                 |
| high      | 15 % - 25 % | Investigate dispatch and loader uptime     |
| critical  | > 25 %      | Take immediate action (reroute trucks)     |

```python
from src.main import HaulTruckAnalyzer
from src.queue_time_analyzer import analyze_queue_time

analyzer = HaulTruckAnalyzer()
import pandas as pd
raw = pd.read_csv("demo/sample_data.csv")
enriched = analyzer.enrich(analyzer.preprocess(raw))
report = analyze_queue_time(enriched)
print(report.fleet_queue_ratio, report.worst_truck)
```

---

## Sample Data

`demo/sample_data.csv` (20 rows, 5 trucks, 3 shovels, day + night shifts):

| Column            | Description                                        |
|-------------------|----------------------------------------------------|
| `cycle_id`        | Unique identifier per cycle                        |
| `truck_id`        | Truck identifier (e.g. `HT101`)                    |
| `truck_type`      | Model (CAT 777E, Komatsu HD785, Volvo A60H)        |
| `shovel_id`       | Loading unit identifier                            |
| `timestamp_start` | ISO-8601 cycle start timestamp                     |
| `load_time_min`   | Time spent loading at the shovel (minutes, 2-5)    |
| `haul_time_min`   | Loaded haul travel time (minutes, 8-18)            |
| `dump_time_min`   | Time at dump/crusher (minutes, 1-3)                |
| `return_time_min` | Empty return travel time (minutes, 6-14)           |
| `queue_time_min`  | Time waiting in shovel queue (minutes, 0-6)        |
| `payload_tonnes`  | Payload per cycle (tonnes, 70-100)                 |
| `shift`           | `Day` or `Night`                                   |
| `material`        | `coal` or `overburden`                             |

Truck models represented:

| Truck ID | Model          | Nominal Payload |
|----------|----------------|-----------------|
| HT101    | CAT 777E       | ~91 t           |
| HT102    | Komatsu HD785  | ~91 t           |
| HT103    | Volvo A60H     | ~60 t           |
| HT104    | CAT 777E       | ~91 t           |
| HT105    | Komatsu HD785  | ~91 t           |

---

## Methodology Notes

- **Cycle time** is the sum of the five stage minutes: load + haul + dump +
  return + queue. Missing stage cells are treated as zero; negative cells
  are clamped to zero in `preprocess` and excluded from the decomposition
  statistics.
- **Productivity** is `payload_tonnes / cycle_min * 60`, expressed as t/h.
  Degenerate denominators (zero or negative cycle time) yield 0.0 rather
  than raising.
- **Match factor** follows Atkinson (1992): `MF = (n_trucks *
  loader_cycle_time) / (truck_cycle_time * n_passes)`. MF < 0.9 means the
  loader is the bottleneck; MF > 1.1 means trucks are the bottleneck.
- **Bottleneck (per cycle)** is the stage with the largest single-row time
  component.
- **Bottleneck (fleet)** is the stage with the largest mean time across all
  cycles; see `cycle_decomposition.decompose_cycle`.
- **Queue ratio** is `sum(queue_min) / sum(cycle_min)` per truck; thresholds
  for severity buckets follow industry rules of thumb for open-pit
  dispatch.
- **Outlier detection** supports IQR (Tukey fence, default multiplier 1.5)
  and z-score (default threshold 3.0); both are robust to constant columns
  and NaN cells.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers 206 cases across 11 test modules. Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
haul-truck-cycle-analyzer/
├── src/
│   ├── __init__.py
│   ├── main.py                            # Core pipeline, KPI helpers, validation
│   ├── cycle_decomposition.py             # Per-stage stats + fleet bottleneck (new)
│   ├── fleet_match_factor_calculator.py   # Loader/truck balance KPI
│   ├── queue_time_analyzer.py             # Per-truck queue severity
│   ├── outlier_filter.py                  # IQR / z-score outlier removal
│   └── data_generator.py                  # Synthetic data for examples
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py                   # 50+ unit + integration cases
│   ├── test_cycle_decomposition.py        # 40+ cases for the new module
│   ├── test_fleet_match_factor_calculator.py
│   ├── test_queue_time_analyzer.py
│   └── test_outlier_filter.py
├── demo/
│   └── sample_data.csv                    # 20-row open-pit sample dataset
├── examples/
│   └── basic_usage.py
├── data/                                  # Drop real data files here (gitignored)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## License

MIT License -- free to use, modify, and distribute.
