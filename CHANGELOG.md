# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-04-20

### Added

- **`src/cycle_decomposition.py`**: New module that decomposes each haul truck
  cycle into its five canonical stages (load, haul, dump, return, queue) and
  reports fleet-level descriptive statistics plus the dominant stage.
  - `decompose_cycle(df)`: returns an immutable `CycleDecompositionReport`
    with per-stage `StageStats` (mean, median, p95, min, max, std,
    share_of_cycle) and the fleet-wide `bottleneck_stage` / `bottleneck_share`.
  - `rank_stages_by_median(df)`: alternative ranking robust to long-tailed
    stage distributions.
  - Accepts both the standard schema (`loading_time_min`, ...) and the
    timestamp schema (`load_time_min`, ...) column aliases transparently.
  - Edge cases: empty DataFrame, all-NaN rows, missing stage columns,
    negative durations (excluded rather than clamped to keep stats honest),
    zero-valued stages, single-row input, duplicate truck IDs.
  - Frozen dataclass result objects; input DataFrame is never mutated.
  - `CycleDecompositionReport.to_dataframe()` for dashboard export.
- **`tests/test_cycle_decomposition.py`**: 41 pytest cases covering happy
  path, match-factor math sanity (verifying M=1 for a balanced setup),
  fleet-level bottleneck detection, all edge cases listed above,
  immutability guarantees, DataFrame round-trip, and end-to-end integration
  against the refreshed `demo/sample_data.csv`.
- **`demo/sample_data.csv`**: Regenerated 20-row open-pit sample with
  smaller-class trucks (CAT 777E, Komatsu HD785, Volvo A60H) and a new
  schema: `cycle_id, truck_id, truck_type, shovel_id, timestamp_start,
  load_time_min, haul_time_min, dump_time_min, return_time_min,
  queue_time_min, payload_tonnes, shift, material`.
- **`src/__init__.py`**: Exposes the new cycle_decomposition API at package
  level via `__all__`.

### Changed

- **README**: Rewritten with clearer installation and quickstart flow, a new
  "Cycle Decomposition" section, updated sample-data column reference, and
  a methodology-notes section summarising every KPI formula.
- **`tests/test_queue_time_analyzer.py::TestSampleDataIntegration`**: Enrich
  the raw sample CSV through `HaulTruckAnalyzer.preprocess/enrich` before
  analysis, since the refreshed sample stores only stage times (no
  precomputed `total_cycle_min`).

## [Unreleased] - 2026-04-19

### Added

- **`src/outlier_filter.py`**: New module to identify and remove abnormal
  cycle records before computing KPIs.
  - `filter_outliers(df, column, method, ...)`: returns a new DataFrame with
    outlier rows removed plus an immutable `OutlierReport`.
  - `flag_outliers(df, ...)`: non-destructive variant that adds a boolean
    `is_outlier` column instead of dropping rows.
  - Supports both IQR (Tukey fence, default multiplier 1.5) and z-score
    (default threshold 3.0) detection strategies.
  - Handles edge cases: empty DataFrame, constant column (zero variance),
    NaN cells, non-numeric columns, unknown method names, and invalid
    threshold parameters.
  - Never mutates the input DataFrame; preserves index and column order.
- **`tests/test_outlier_filter.py`**: 16 pytest cases covering happy path
  for both methods, input validation, immutability, NaN handling, frozen
  dataclass guarantees, and constant-column degeneracy.

## [0.2.0] - 2026-04-18

### Added

- **`src/queue_time_analyzer.py`**: New module that quantifies per-truck and
  fleet-wide queue-time burden as a percentage of cycle time and tags each
  truck with a severity bucket (`low`, `moderate`, `high`, `critical`).
  - `analyze_queue_time(df, ...)`: aggregates queue and cycle minutes per
    truck, returning an immutable `QueueTimeReport` sorted worst-first.
  - `classify_queue_severity(queue_ratio)`: pure helper mapping a queue ratio
    in `[0, 1]` to a severity bucket; raises `ValueError` on out-of-range or
    NaN input.
  - `TruckQueueStats` and `QueueTimeReport` frozen dataclasses for type-safe,
    immutable result representation.
  - Graceful handling of empty DataFrames, missing columns, zero/negative
    cycle times, missing stage time, and pathological `queue > cycle` rows
    (the latter clamped to cycle for an upper bound of 1.0).
  - Fallback from `total_cycle_min` to `computed_cycle_min` mirroring the
    rest of the package.
  - `QueueTimeReport.to_dataframe()` for downstream CSV/Excel export and BI
    dashboards.
- **`tests/test_queue_time_analyzer.py`**: 40 pytest cases covering happy
  path, severity boundary classification, edge cases (empty DataFrame,
  single truck, missing stage time, all-same-route fleets, NaN values, queue
  greater than cycle), immutability, determinism, and integration against
  the bundled sample dataset.
- **README**: Added "New: Queue Time Analyzer" section with severity bucket
  table, step-by-step usage, and standalone classifier example.
- **`src/__init__.py`**: Exposes the new public API at the package level via
  `__all__`.

## [Unreleased] - 2026-04-17

### Added

- **`src/fleet_match_factor_calculator.py`**: New module implementing the
  loader-to-truck match factor (MF) KPI for open-pit mining operations.
  - `compute_match_factor(n_trucks, truck_cycle_time_min, loader_cycle_time_min, n_passes)`:
    pure function returning the dimensionless MF, with full input validation.
  - `calculate_fleet_match_factor(df, loader_cycle_time_min, ...)`: DataFrame-level
    calculator that groups cycles by pit, counts distinct trucks, computes mean
    cycle time per pit, and returns an immutable `MatchFactorReport`.
  - `PitMatchResult` and `MatchFactorReport` frozen dataclasses for type-safe,
    immutable result representation.
  - Automatic condition labelling: `"under-trucked"` (MF < 0.90),
    `"balanced"` (0.90–1.10), `"over-trucked"` (MF > 1.10).
  - `MatchFactorReport.to_dataframe()` for downstream CSV/Excel export.
  - Graceful handling of empty DataFrames, missing columns, and zero/invalid
    cycle times (invalid rows skipped, not rejected).
- **`tests/test_fleet_match_factor_calculator.py`**: 30 pytest tests covering
  happy path, classification thresholds, edge cases (empty DF, single truck,
  zero payload cycles, multi-pit weighting), immutability, determinism,
  parametrized condition boundaries, DataFrame round-trip, and invalid-input
  error handling.
- **README**: Added "New: Fleet Match Factor Calculator" section with
  step-by-step usage and standalone examples.

## [0.2.0] - 2026-04-16

### Added

- **Input validation** (`src/main.py`):
  - `_validate_no_missing_truck_ids`: raises `ValueError` for null/blank truck IDs
  - `_validate_timestamps_ordered`: raises `ValueError` when `timestamp_end <= timestamp_start`
  - `_validate_payload_non_negative`: raises `ValueError` for negative `payload_tonnes` or `load_tonnes`
  - `_validate_cycle_times_non_negative`: raises `ValueError` for negative time components (configurable via `strict_validation`)
  - `_validate_no_duplicate_truck_trip`: raises `ValueError` for duplicate `(truck_id, trip_id)` pairs
- **Timestamp schema support**: `HaulTruckAnalyzer.preprocess` now normalises column aliases:
  - `load_time_min` -> `loading_time_min`
  - `haul_time_min` -> `hauling_time_min`
  - `dump_time_min` -> `dumping_time_min`
  - `payload_tonnes` -> `load_tonnes`
  - `haul_distance_km` -> `distance_km`
- **Negative payload clamping** in `preprocess` (in addition to existing time/distance clamping)
- **`fleet_summary` fallback** to `computed_cycle_min` when `total_cycle_min` is absent
- **Comprehensive docstrings** across all public functions and the `HaulTruckAnalyzer` class, including `Parameters`, `Returns`, `Raises`, and `Examples` sections
- **Expanded unit tests** (`tests/test_analyzer.py`): 50+ test cases across 11 test classes covering:
  - Cycle time summation edge cases (null values, large realistic inputs)
  - Productivity edge cases (Liebherr T 284, Komatsu HD785 scale payloads)
  - All five bottleneck phases (loading, hauling, dumping, return, queue)
  - Utilization capping and zero-division safety
  - All new validation helper functions
  - Column alias normalisation
  - Timestamp schema end-to-end pipeline
  - `to_dataframe` serialisation including dot-notation flattening
- **Revised `demo/sample_data.csv`** (20 rows): realistic open-pit coal mining data with columns `truck_id`, `trip_id`, `timestamp_start`, `timestamp_end`, `load_time_min`, `haul_time_min`, `dump_time_min`, `return_time_min`, `payload_tonnes`, `haul_distance_km`, `material_type`, `pit_name`, `queue_time_min`, `total_cycle_min`, `shift`, `date`; models include CAT 793F, Komatsu 830E, Liebherr T 284
- **Improved README**: badges, Quick Start section, step-by-step pipeline, timestamp-schema example, Sample Output table, fleet model table, and expanded project structure reference

### Changed

- `HaulTruckAnalyzer.validate` now calls all five structural validation helpers in sequence
- `HaulTruckAnalyzer.preprocess` applies alias renaming before clamping, ensuring both column schemas produce identical internal representation
- `fleet_summary` rename map extended to handle `computed_cycle_min` aggregation columns

### Fixed

- `fleet_summary` now correctly handles DataFrames where only `computed_cycle_min` (not `total_cycle_min`) is present, preventing missing `cycle_count` / `avg_cycle_min` columns in the output

## [0.1.0] - 2026-01-01

### Added

- Initial release with `HaulTruckAnalyzer` class
- CSV and Excel data loading
- Basic KPI computation: cycle time, productivity, bottleneck detection
- `fleet_summary` aggregation
- `data_generator.py` for synthetic data generation
- Initial README and requirements
