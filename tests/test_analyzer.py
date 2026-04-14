"""
Unit tests for HaulTruckAnalyzer and its pure helper functions.

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
    """Minimal valid DataFrame for analyzer tests."""
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

    def test_zero_available_time_returns_zero(self) -> None:
        """Zero available time must not cause a division by zero error."""
        assert compute_utilization(100.0, 0.0) == 0.0

    def test_negative_available_time_returns_zero(self) -> None:
        """Negative available time is invalid; should return 0.0."""
        assert compute_utilization(100.0, -60.0) == 0.0

    def test_zero_active_time_returns_zero(self) -> None:
        """Zero active time means the truck was idle the entire shift."""
        assert compute_utilization(0.0, 720.0) == 0.0


# ---------------------------------------------------------------------------
# Test 5 – Edge cases: zero times, negative distances
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
        # Original must be unchanged
        assert original["loading_time_min"].iloc[0] == original_value


# ---------------------------------------------------------------------------
# Test 6 – Fleet summary statistics
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
        # T1 has 2 cycles, T2 has 1
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


# ---------------------------------------------------------------------------
# Test 7 – Full analyze pipeline
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


# ---------------------------------------------------------------------------
# Test 8 – to_dataframe export helper
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
