# Changelog

All notable changes to this project are documented in this file.

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
