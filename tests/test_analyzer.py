"""
Unit tests for HaulTruckAnalyzer and its pure helper functions.

Test classes:
    TestComputeCycleTime        - cycle time summation helper
    TestComputeProductivity     - tonnes-per-hour productivity helper
    TestIdentifyBottleneck      - dominant phase detection helper
    TestComputeUtilization      - utilization rate helper
    TestValidation              - DataFrame validation rules
    TestEdgeCases               - edge cases via HaulTruckAnalyzer
    TestPreprocessNormalisation - column alias normalisation
    TestFleetSummary            - fleet-level aggregation
    TestAnalyzePipeline         - end-to-end analyze() method
    TestToDataframe             - result serialisation helper
    TestTimestampSchema         - timestamp-schema (trip_id / timestamps) support

Run with:
    pytest tests/test_analyzer.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup – allow imports without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import (
    HaulTruckAnalyzer,
    compute_cycle_time,
    compute_productivity,
    compute_utilization,
    fleet_summary,
    identify_bottleneck,
    _validate_no_missing_truck_ids,
    _validate_timestamps_ordered,
    _validate_payload_non_negative,
    _validate_cycle_times_non_negative,
    _validate_no_duplicate_truck_trip,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_row() -> pd.Series:
    """A representative cycle row with all time columns populated."""
    return pd.Series(
        {
            "loading_time_min": 8.0,
            "hauling_time_min": 18.5,
            "dumping_time_min": 3.0,
            "return_time_min": 16.0,
            "queue_time_min": 5.0,
            "load_tonnes": 220.0,
            "total_cycle_min": 50.5,
        }
    )


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Minimal valid DataFrame using the standard schema."""
    return pd.DataFrame(
        {
            "cycle_id": ["C1", "C2", "C3"],
            "truck_id": ["T1", "T1", "T2"],
            "load_tonnes": [200.0, 210.0, 180.0],
            "loading_time_min": [8.0, 9.0, 7.5],
            "hauling_time_min": [18.0, 20.0, 22.0],
            "dumping_time_min": [3.0, 3.5, 2.8],
            "return_time_min": [16.0, 17.0, 19.0],
            "queue_time_min": [4.0, 5.5, 6.0],
            "total_cycle_min": [49.0, 55.0, 57.3],
            "distance_km": [4.2, 4.2, 4.8],
        }
    )


@pytest.fixture()
def single_truck_df() -> pd.DataFrame:
    """DataFrame representing a single-truck fleet."""
    return pd.DataFrame(
        {
            "cycle_id": ["C1", "C2"],
            "truck_id": ["T1", "T1"],
            "load_tonnes": [215.0, 220.0],
            "loading_time_min": [8.5, 8.0],
            "hauling_time_min": [19.0, 18.5],
            "dumping_time_min": [3.2, 3.0],
            "return_time_min": [16.5, 16.0],
            "queue_time_min": [4.5, 4.0],
            "total_cycle_min": [51.7, 49.5],
            "distance_km": [4.2, 4.2],
        }
    )


@pytest.fixture()
def timestamp_schema_df() -> pd.DataFrame:
    """Valid DataFrame in the timestamp schema (load_time_min aliases)."""
    return pd.DataFrame(
        {
            "truck_id": ["TRK01", "TRK01", "TRK02"],
            "trip_id": ["T001", "T002", "T003"],
            "timestamp_start": [
                "2026-01-05 06:00:00",
                "2026-01-05 07:05:00",
                "2026-01-05 06:05:00",
            ],
            "timestamp_end": [
                "2026-01-05 06:50:18",
                "2026-01-05 07:54:18",
                "2026-01-05 06:59:54",
            ],
            "load_time_min": [8.2, 7.8, 7.5],
            "haul_time_min": [18.5, 18.0, 20.1],
            "dump_time_min": [3.1, 2.9, 2.8],
            "return_time_min": [16.0, 15.5, 18.3],
            "payload_tonnes": [220.5, 225.0, 185.0],
            "haul_distance_km": [4.2, 4.2, 4.8],
            "material_type": ["Coal", "Coal", "Coal"],
            "pit_name": ["North Pit", "North Pit", "East Pit"],
        }
    )


@pytest.fixture()
def analyzer() -> HaulTruckAnalyzer:
    """Default analyzer instance."""
    return HaulTruckAnalyzer()


# ---------------------------------------------------------------------------
# Test 1 – Cycle time calculation
# ---------------------------------------------------------------------------


class TestComputeCycleTime:
    """Tests for the compute_cycle_time helper."""

    def test_normal_row_sums_all_time_columns(self, sample_row: pd.Series) -> None:
        """All five time columns should be summed correctly."""
        result = compute_cycle_time(sample_row)
        expected = 8.0 + 18.5 + 3.0 + 16.0 + 5.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_times_returns_zero(self) -> None:
        """All-zero time columns must yield 0.0 without error."""
        row = pd.Series(
            {
                "loading_time_min": 0.0,
                "hauling_time_min": 0.0,
                "dumping_time_min": 0.0,
                "return_time_min": 0.0,
                "queue_time_min": 0.0,
            }
        )
        assert compute_cycle_time(row) == 0.0

    def test_negative_values_clamped_to_zero(self) -> None:
        """Negative time components should be treated as zero."""
        row = pd.Series(
            {
                "loading_time_min": -5.0,
                "hauling_time_min": 18.0,
                "dumping_time_min": 3.0,
                "return_time_min": 16.0,
            }
        )
        result = compute_cycle_time(row)
        assert result == pytest.approx(37.0, rel=1e-4)

    def test_missing_optional_queue_column(self) -> None:
        """queue_time_min is optional; its absence should not raise."""
        row = pd.Series(
            {
                "loading_time_min": 8.0,
                "hauling_time_min": 18.0,
                "dumping_time_min": 3.0,
                "return_time_min": 16.0,
            }
        )
        result = compute_cycle_time(row)
        assert result == pytest.approx(45.0, rel=1e-4)

    def test_no_time_columns_returns_zero(self) -> None:
        """Row with no recognised time columns should return 0.0."""
        row = pd.Series({"truck_id": "T1", "load_tonnes": 200.0})
        assert compute_cycle_time(row) == 0.0

    def test_null_time_values_treated_as_zero(self) -> None:
        """NaN time components should contribute 0.0 to the total."""
        row = pd.Series(
            {
                "loading_time_min": float("nan"),
                "hauling_time_min": 18.0,
                "dumping_time_min": 3.0,
                "return_time_min": 16.0,
            }
        )
        result = compute_cycle_time(row)
        assert result == pytest.approx(37.0, rel=1e-4)

    def test_large_realistic_cycle_time(self) -> None:
        """A Liebherr T 284 hauling over 5 km should produce a plausible total."""
        row = pd.Series(
            {
                "loading_time_min": 11.0,
                "hauling_time_min": 23.1,
                "dumping_time_min": 4.2,
                "return_time_min": 21.5,
                "queue_time_min": 8.0,
            }
        )
        result = compute_cycle_time(row)
        assert result == pytest.approx(67.8, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 2 – Productivity metrics (tonnes / hour)
# ---------------------------------------------------------------------------


class TestComputeProductivity:
    """Tests for the compute_productivity helper."""

    def test_standard_productivity(self) -> None:
        """220 tonnes / 50 min = 264 t/h."""
        result = compute_productivity(220.0, 50.0)
        expected = (220.0 / 50.0) * 60.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_cycle_time_returns_zero(self) -> None:
        """Division by zero must be handled gracefully."""
        assert compute_productivity(220.0, 0.0) == 0.0

    def test_negative_cycle_time_returns_zero(self) -> None:
        """Negative cycle time is invalid and should return 0.0."""
        assert compute_productivity(220.0, -10.0) == 0.0

    def test_zero_load_tonnes_returns_zero(self) -> None:
        """Zero payload yields zero productivity."""
        assert compute_productivity(0.0, 50.0) == 0.0

    def test_negative_load_tonnes_returns_zero(self) -> None:
        """Negative tonnes is nonsensical; function must return 0.0."""
        assert compute_productivity(-50.0, 50.0) == 0.0

    def test_liebherr_productivity_range(self) -> None:
        """Liebherr T 284 payload of ~290 t with ~65 min cycle is realistic."""
        result = compute_productivity(290.0, 65.0)
        # Expect roughly 267 t/h
        assert 250.0 < result < 290.0

    def test_komatsu_hd785_productivity(self) -> None:
        """Komatsu HD785 at 91 t payload and 55 min cycle produces ~99 t/h."""
        result = compute_productivity(91.0, 55.0)
        assert result == pytest.approx((91.0 / 55.0) * 60.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 3 – Bottleneck identification
# ---------------------------------------------------------------------------


class TestIdentifyBottleneck:
    """Tests for the identify_bottleneck helper."""

    def test_hauling_is_dominant(self, sample_row: pd.Series) -> None:
        """With hauling_time_min=18.5 as max, bottleneck should be hauling."""
        assert identify_bottleneck(sample_row) == "hauling_time_min"

    def test_queue_is_dominant_when_largest(self) -> None:
        """queue_time_min dominates when it has the highest value."""
        row = pd.Series(
            {
                "loading_time_min": 5.0,
                "hauling_time_min": 10.0,
                "dumping_time_min": 2.0,
                "return_time_min": 9.0,
                "queue_time_min": 25.0,
            }
        )
        assert identify_bottleneck(row) == "queue_time_min"

    def test_loading_is_dominant(self) -> None:
        """loading_time_min dominates when it exceeds all other phases."""
        row = pd.Series(
            {
                "loading_time_min": 30.0,
                "hauling_time_min": 18.0,
                "dumping_time_min": 3.0,
                "return_time_min": 16.0,
            }
        )
        assert identify_bottleneck(row) == "loading_time_min"

    def test_return_is_dominant(self) -> None:
        """return_time_min dominates on long empty-return routes."""
        row = pd.Series(
            {
                "loading_time_min": 8.0,
                "hauling_time_min": 18.0,
                "dumping_time_min": 3.0,
                "return_time_min": 35.0,
            }
        )
        assert identify_bottleneck(row) == "return_time_min"

    def test_all_zero_returns_unknown(self) -> None:
        """All-zero times have no bottleneck; returns 'unknown'."""
        row = pd.Series(
            {
                "loading_time_min": 0.0,
                "hauling_time_min": 0.0,
                "dumping_time_min": 0.0,
                "return_time_min": 0.0,
            }
        )
        assert identify_bottleneck(row) == "unknown"

    def test_no_time_columns_returns_unknown(self) -> None:
        """Row without any time columns returns 'unknown'."""
        row = pd.Series({"truck_id": "T1"})
        assert identify_bottleneck(row) == "unknown"


# ---------------------------------------------------------------------------
# Test 4 – Utilization rate
# ---------------------------------------------------------------------------


class TestComputeUtilization:
    """Tests for the compute_utilization helper."""

    def test_standard_utilization(self) -> None:
        """600 active min out of 720 available = 83.33 %."""
        result = compute_utilization(600.0, 720.0)
        assert result == pytest.approx(83.3333, rel=1e-3)

    def test_full_utilization_capped_at_100(self) -> None:
        """Active time equal to available time must return exactly 100.0."""
        assert compute_utilization(720.0, 720.0) == pytest.approx(100.0)

    def test_over_utilization_capped_at_100(self) -> None:
        """Active time exceeding available must be capped at 100.0."""
        assert compute_utilization(800.0, 720.0) == pytest.approx(100.0)

    def test_zero_available_time_returns_zero(self) -> None:
        """Zero available time must not cause a division by zero error."""
        assert compute_utilization(100.0, 0.0) == 0.0

    def test_negative_available_time_returns_zero(self) -> None:
        """Negative available time is invalid; should return 0.0."""
        assert compute_utilization(100.0, -60.0) == 0.0

    def test_zero_active_time_returns_zero(self) -> None:
        """Zero active time means the truck was idle the entire shift."""
        assert compute_utilization(0.0, 720.0) == 0.0

    def test_half_shift_utilization(self) -> None:
        """360 min active out of 720 available equals exactly 50.0 %."""
        assert compute_utilization(360.0, 720.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Test 5 – Input validation helpers
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for low-level validation helper functions and HaulTruckAnalyzer.validate."""

    def test_validate_empty_df_raises(self, analyzer: HaulTruckAnalyzer) -> None:
        """Empty DataFrame must raise ValueError with 'empty' in message."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.validate(pd.DataFrame())

    def test_validate_missing_required_columns_raises(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """DataFrame missing all required time columns must raise ValueError."""
        df = pd.DataFrame({"truck_id": ["T1"], "load_tonnes": [200.0]})
        with pytest.raises(ValueError, match="Missing required time columns"):
            analyzer.validate(df)

    def test_validate_missing_truck_id_raises(self) -> None:
        """Null truck_id values must raise ValueError."""
        df = pd.DataFrame(
            {
                "truck_id": [None, "T2"],
                "loading_time_min": [8.0, 8.0],
                "hauling_time_min": [18.0, 18.0],
                "dumping_time_min": [3.0, 3.0],
                "return_time_min": [16.0, 16.0],
            }
        )
        with pytest.raises(ValueError, match="Missing or blank truck_id"):
            _validate_no_missing_truck_ids(df)

    def test_validate_blank_string_truck_id_raises(self) -> None:
        """Blank-string truck_id (spaces only) must also be rejected."""
        df = pd.DataFrame(
            {
                "truck_id": ["   ", "T2"],
                "loading_time_min": [8.0, 8.0],
            }
        )
        with pytest.raises(ValueError, match="Missing or blank truck_id"):
            _validate_no_missing_truck_ids(df)

    def test_validate_timestamps_out_of_order_raises(self) -> None:
        """timestamp_end before timestamp_start must raise ValueError."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "timestamp_start": ["2026-01-05 07:00:00"],
                "timestamp_end": ["2026-01-05 06:00:00"],  # end before start
            }
        )
        with pytest.raises(ValueError, match="timestamp_end is not after timestamp_start"):
            _validate_timestamps_ordered(df)

    def test_validate_equal_timestamps_raises(self) -> None:
        """timestamp_end equal to timestamp_start (zero duration) must raise."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "timestamp_start": ["2026-01-05 06:00:00"],
                "timestamp_end": ["2026-01-05 06:00:00"],
            }
        )
        with pytest.raises(ValueError, match="timestamp_end is not after timestamp_start"):
            _validate_timestamps_ordered(df)

    def test_validate_timestamps_in_order_passes(self) -> None:
        """Correctly ordered timestamps must pass without raising."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "timestamp_start": ["2026-01-05 06:00:00"],
                "timestamp_end": ["2026-01-05 06:50:00"],
            }
        )
        _validate_timestamps_ordered(df)  # must not raise

    def test_validate_negative_payload_raises(self) -> None:
        """Negative payload_tonnes values must raise ValueError."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "payload_tonnes": [-100.0],
            }
        )
        with pytest.raises(ValueError, match="Negative payload values"):
            _validate_payload_non_negative(df)

    def test_validate_negative_load_tonnes_raises(self) -> None:
        """Negative load_tonnes values must raise ValueError."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "load_tonnes": [-50.0],
            }
        )
        with pytest.raises(ValueError, match="Negative payload values"):
            _validate_payload_non_negative(df)

    def test_validate_zero_payload_passes(self) -> None:
        """Zero payload must be allowed (truck ran empty – valid edge case)."""
        df = pd.DataFrame({"truck_id": ["T1"], "load_tonnes": [0.0]})
        _validate_payload_non_negative(df)  # must not raise

    def test_validate_negative_cycle_time_raises_in_strict_mode(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """Negative cycle times must raise ValueError when strict_validation=True."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "loading_time_min": [-5.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
            }
        )
        with pytest.raises(ValueError, match="Negative cycle time values"):
            _validate_cycle_times_non_negative(df)

    def test_validate_passes_for_valid_df(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """validate() on a clean DataFrame must return True without raising."""
        assert analyzer.validate(minimal_df) is True

    def test_validate_duplicate_truck_trip_raises(self) -> None:
        """Duplicate (truck_id, trip_id) pairs must raise ValueError."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T1"],
                "trip_id": ["TR01", "TR01"],
                "loading_time_min": [8.0, 8.0],
            }
        )
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_no_duplicate_truck_trip(df)

    def test_validate_no_duplicate_truck_trip_passes(self) -> None:
        """Unique (truck_id, trip_id) pairs must pass without error."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T1", "T2"],
                "trip_id": ["TR01", "TR02", "TR01"],
            }
        )
        _validate_no_duplicate_truck_trip(df)  # must not raise


# ---------------------------------------------------------------------------
# Test 6 – Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Integration-level edge case tests using HaulTruckAnalyzer."""

    def test_all_zero_cycle_times_do_not_raise(self, analyzer: HaulTruckAnalyzer) -> None:
        """DataFrame with all-zero time columns should not raise any error."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "load_tonnes": [200.0],
                "loading_time_min": [0.0],
                "hauling_time_min": [0.0],
                "dumping_time_min": [0.0],
                "return_time_min": [0.0],
                "queue_time_min": [0.0],
                "total_cycle_min": [0.0],
            }
        )
        result = analyzer.analyze(df)
        assert result["total_records"] == 1

    def test_negative_distance_clamped_to_zero(self, analyzer: HaulTruckAnalyzer) -> None:
        """Negative distance values must be clamped to 0 during preprocessing."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "load_tonnes": [200.0],
                "loading_time_min": [8.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
                "distance_km": [-5.0],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert (preprocessed["distance_km"] >= 0.0).all()

    def test_negative_haul_distance_km_alias_clamped(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """Negative haul_distance_km (alias) must also be clamped to 0."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "load_time_min": [8.0],
                "haul_time_min": [18.0],
                "dump_time_min": [3.0],
                "return_time_min": [16.0],
                "haul_distance_km": [-3.5],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert preprocessed["distance_km"].iloc[0] == 0.0

    def test_missing_cycles_in_df(self, analyzer: HaulTruckAnalyzer) -> None:
        """Empty DataFrame should raise ValueError on validate."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.validate(pd.DataFrame())

    def test_preprocess_does_not_mutate_input(self) -> None:
        """preprocess must return a new DataFrame, not mutate the original."""
        analyzer = HaulTruckAnalyzer()
        original = pd.DataFrame(
            {
                "loading_time_min": [-1.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
            }
        )
        original_value = original["loading_time_min"].iloc[0]
        _ = analyzer.preprocess(original)
        assert original["loading_time_min"].iloc[0] == original_value

    def test_single_row_df_does_not_raise(self, analyzer: HaulTruckAnalyzer) -> None:
        """A single-row DataFrame must be processed without error."""
        df = pd.DataFrame(
            {
                "truck_id": ["TRK01"],
                "load_tonnes": [220.5],
                "loading_time_min": [8.2],
                "hauling_time_min": [18.5],
                "dumping_time_min": [3.1],
                "return_time_min": [16.0],
            }
        )
        result = analyzer.analyze(df)
        assert result["total_records"] == 1

    def test_non_strict_mode_allows_negative_times(self) -> None:
        """With strict_validation=False, negative times should not raise on validate."""
        analyzer = HaulTruckAnalyzer(config={"strict_validation": False})
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "loading_time_min": [-5.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
            }
        )
        # Should not raise
        result = analyzer.validate(df)
        assert result is True


# ---------------------------------------------------------------------------
# Test 7 – Preprocessing and column alias normalisation
# ---------------------------------------------------------------------------


class TestPreprocessNormalisation:
    """Tests for column alias handling in preprocess()."""

    def test_load_time_min_renamed_to_loading_time_min(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """load_time_min alias must be renamed to loading_time_min."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "load_time_min": [8.0],
                "haul_time_min": [18.0],
                "dump_time_min": [3.0],
                "return_time_min": [16.0],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert "loading_time_min" in preprocessed.columns
        assert "load_time_min" not in preprocessed.columns

    def test_payload_tonnes_renamed_to_load_tonnes(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """payload_tonnes alias must be renamed to load_tonnes."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "loading_time_min": [8.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
                "payload_tonnes": [220.5],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert "load_tonnes" in preprocessed.columns
        assert "payload_tonnes" not in preprocessed.columns

    def test_haul_distance_km_renamed_to_distance_km(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """haul_distance_km alias must be renamed to distance_km."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "loading_time_min": [8.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
                "haul_distance_km": [4.2],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert "distance_km" in preprocessed.columns
        assert "haul_distance_km" not in preprocessed.columns

    def test_fully_empty_rows_dropped(self, analyzer: HaulTruckAnalyzer) -> None:
        """Fully empty rows (all NaN) must be dropped during preprocessing."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1", None],
                "loading_time_min": [8.0, None],
                "hauling_time_min": [18.0, None],
                "dumping_time_min": [3.0, None],
                "return_time_min": [16.0, None],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert len(preprocessed) == 1

    def test_negative_payload_tonnes_clamped_in_preprocess(
        self, analyzer: HaulTruckAnalyzer
    ) -> None:
        """Negative payload_tonnes values must be clamped to 0 after preprocessing."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "loading_time_min": [8.0],
                "hauling_time_min": [18.0],
                "dumping_time_min": [3.0],
                "return_time_min": [16.0],
                "payload_tonnes": [-50.0],
            }
        )
        preprocessed = analyzer.preprocess(df)
        assert preprocessed["load_tonnes"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Test 8 – Fleet summary statistics
# ---------------------------------------------------------------------------


class TestFleetSummary:
    """Tests for the fleet_summary aggregation function."""

    def test_returns_one_row_per_truck(self, minimal_df: pd.DataFrame) -> None:
        """Result should have exactly one row per unique truck_id."""
        analyzer = HaulTruckAnalyzer()
        enriched = analyzer.enrich(analyzer.preprocess(minimal_df))
        summary = fleet_summary(enriched)
        assert len(summary) == minimal_df["truck_id"].nunique()

    def test_cycle_count_correct(self, minimal_df: pd.DataFrame) -> None:
        """Cycle count per truck should match raw row counts in the input."""
        analyzer = HaulTruckAnalyzer()
        enriched = analyzer.enrich(analyzer.preprocess(minimal_df))
        summary = fleet_summary(enriched)
        assert summary.loc["T1", "cycle_count"] == 2
        assert summary.loc["T2", "cycle_count"] == 1

    def test_single_truck_fleet(self, single_truck_df: pd.DataFrame) -> None:
        """A fleet of one truck should produce a one-row summary without error."""
        analyzer = HaulTruckAnalyzer()
        enriched = analyzer.enrich(analyzer.preprocess(single_truck_df))
        summary = fleet_summary(enriched)
        assert len(summary) == 1
        assert summary.index[0] == "T1"

    def test_empty_dataframe_returns_empty_summary(self) -> None:
        """fleet_summary on an empty DataFrame must return an empty DataFrame."""
        result = fleet_summary(pd.DataFrame())
        assert result.empty

    def test_total_tonnes_summed_correctly(self, minimal_df: pd.DataFrame) -> None:
        """Total tonnes for T1 should equal the sum of T1 rows in the input."""
        analyzer = HaulTruckAnalyzer()
        enriched = analyzer.enrich(analyzer.preprocess(minimal_df))
        summary = fleet_summary(enriched)
        expected_t1_tonnes = minimal_df[minimal_df["truck_id"] == "T1"]["load_tonnes"].sum()
        assert summary.loc["T1", "total_tonnes"] == pytest.approx(expected_t1_tonnes, rel=1e-3)

    def test_avg_cycle_min_is_positive(self, minimal_df: pd.DataFrame) -> None:
        """Average cycle time in the summary must be positive for all trucks."""
        analyzer = HaulTruckAnalyzer()
        enriched = analyzer.enrich(analyzer.preprocess(minimal_df))
        summary = fleet_summary(enriched)
        assert (summary["avg_cycle_min"] > 0).all()

    def test_avg_productivity_tph_positive(self, minimal_df: pd.DataFrame) -> None:
        """Average productivity must be positive when payload is non-zero."""
        analyzer = HaulTruckAnalyzer()
        enriched = analyzer.enrich(analyzer.preprocess(minimal_df))
        summary = fleet_summary(enriched)
        assert (summary["avg_productivity_tph"] > 0).all()


# ---------------------------------------------------------------------------
# Test 9 – Full analyze pipeline
# ---------------------------------------------------------------------------


class TestAnalyzePipeline:
    """End-to-end tests for HaulTruckAnalyzer.analyze."""

    def test_analyze_returns_expected_keys(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """The result dict must include all documented top-level keys."""
        result = analyzer.analyze(minimal_df)
        for key in ("total_records", "columns", "missing_pct"):
            assert key in result, f"Key '{key}' missing from result"

    def test_total_records_matches_input(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """total_records must equal the number of input rows."""
        result = analyzer.analyze(minimal_df)
        assert result["total_records"] == len(minimal_df)

    def test_bottleneck_distribution_present(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """bottleneck_distribution must be a non-empty dict after analysis."""
        result = analyzer.analyze(minimal_df)
        assert "bottleneck_distribution" in result
        assert isinstance(result["bottleneck_distribution"], dict)
        assert len(result["bottleneck_distribution"]) > 0

    def test_productivity_computed_and_positive(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """Mean productivity across cycles must be strictly positive."""
        result = analyzer.analyze(minimal_df)
        assert result["means"].get("productivity_tph", 0.0) > 0.0

    def test_run_pipeline_with_sample_csv(self, analyzer: HaulTruckAnalyzer) -> None:
        """run() on the bundled sample_data.csv must succeed end-to-end."""
        sample_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not sample_path.exists():
            pytest.skip("demo/sample_data.csv not found – skipping integration test")
        result = analyzer.run(str(sample_path))
        assert result["total_records"] == 20

    def test_fleet_summary_present_in_analyze(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """fleet_summary key must be present and non-empty after analyze."""
        result = analyzer.analyze(minimal_df)
        assert "fleet_summary" in result
        assert result["fleet_summary"]

    def test_analyze_with_timestamp_schema(
        self, analyzer: HaulTruckAnalyzer, timestamp_schema_df: pd.DataFrame
    ) -> None:
        """analyze() must succeed on the timestamp schema with alias columns."""
        result = analyzer.analyze(timestamp_schema_df)
        assert result["total_records"] == 3
        assert "bottleneck_distribution" in result

    def test_missing_pct_all_zero_for_clean_data(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """A fully populated DataFrame must report 0% missing for all columns."""
        result = analyzer.analyze(minimal_df)
        for col, pct in result["missing_pct"].items():
            assert pct == 0.0, f"Unexpected missing data in column '{col}'"


# ---------------------------------------------------------------------------
# Test 10 – Timestamp schema support
# ---------------------------------------------------------------------------


class TestTimestampSchema:
    """Tests specifically targeting the timestamp-schema CSV format."""

    def test_full_pipeline_on_sample_csv(self, analyzer: HaulTruckAnalyzer) -> None:
        """Full run() on the new 20-row sample_data.csv must produce 20 records."""
        sample_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not sample_path.exists():
            pytest.skip("demo/sample_data.csv not found")
        result = analyzer.run(str(sample_path))
        assert result["total_records"] == 20

    def test_alias_columns_resolved_after_validate(
        self, analyzer: HaulTruckAnalyzer, timestamp_schema_df: pd.DataFrame
    ) -> None:
        """validate() on a timestamp-schema DataFrame must return True."""
        result = analyzer.validate(timestamp_schema_df)
        assert result is True

    def test_productivity_computed_from_payload_tonnes(
        self, analyzer: HaulTruckAnalyzer, timestamp_schema_df: pd.DataFrame
    ) -> None:
        """productivity_tph must be non-zero when payload_tonnes is present."""
        result = analyzer.analyze(timestamp_schema_df)
        assert result["means"].get("productivity_tph", 0.0) > 0.0

    def test_fleet_summary_has_correct_truck_count(
        self, analyzer: HaulTruckAnalyzer, timestamp_schema_df: pd.DataFrame
    ) -> None:
        """Fleet summary must contain one row per unique truck in the input."""
        result = analyzer.analyze(timestamp_schema_df)
        fleet = result.get("fleet_summary", {})
        # TRK01 (2 trips) and TRK02 (1 trip)
        cycle_counts = fleet.get("cycle_count", {})
        assert len(cycle_counts) == 2

    def test_material_type_column_preserved(
        self, analyzer: HaulTruckAnalyzer, timestamp_schema_df: pd.DataFrame
    ) -> None:
        """Non-numeric metadata columns like material_type must survive preprocessing."""
        preprocessed = analyzer.preprocess(timestamp_schema_df)
        assert "material_type" in preprocessed.columns

    def test_pit_name_column_preserved(
        self, analyzer: HaulTruckAnalyzer, timestamp_schema_df: pd.DataFrame
    ) -> None:
        """pit_name column must survive preprocessing unchanged."""
        preprocessed = analyzer.preprocess(timestamp_schema_df)
        assert "pit_name" in preprocessed.columns


# ---------------------------------------------------------------------------
# Test 11 – to_dataframe export helper
# ---------------------------------------------------------------------------


class TestToDataframe:
    """Tests for the result serialisation helper."""

    def test_returns_dataframe(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """to_dataframe must return a pandas DataFrame."""
        result = analyzer.analyze(minimal_df)
        exported = analyzer.to_dataframe(result)
        assert isinstance(exported, pd.DataFrame)

    def test_contains_metric_and_value_columns(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """Exported DataFrame must have 'metric' and 'value' columns."""
        result = analyzer.analyze(minimal_df)
        exported = analyzer.to_dataframe(result)
        assert "metric" in exported.columns
        assert "value" in exported.columns

    def test_flat_result_produces_single_row(self) -> None:
        """A flat dict with one key should produce exactly one row."""
        analyzer = HaulTruckAnalyzer()
        exported = analyzer.to_dataframe({"total_records": 10})
        assert len(exported) == 1
        assert exported.iloc[0]["metric"] == "total_records"
        assert exported.iloc[0]["value"] == 10

    def test_nested_dict_flattened_with_dot_notation(self) -> None:
        """Nested keys must appear as 'parent.child' metric names."""
        analyzer = HaulTruckAnalyzer()
        result = {"means": {"productivity_tph": 264.0}}
        exported = analyzer.to_dataframe(result)
        assert "means.productivity_tph" in exported["metric"].values

    def test_no_rows_lost_for_complex_result(
        self, analyzer: HaulTruckAnalyzer, minimal_df: pd.DataFrame
    ) -> None:
        """All top-level keys in the result must produce at least one row."""
        result = analyzer.analyze(minimal_df)
        exported = analyzer.to_dataframe(result)
        assert len(exported) >= len(result)
