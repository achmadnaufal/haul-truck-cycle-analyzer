"""Pytest test suite for src/queue_time_analyzer.py.

Covers:
- Happy-path per-truck and fleet-wide aggregation
- Severity classification thresholds
- Edge cases: empty DataFrame, missing columns, zero cycles, single truck,
  all-same-route (single pit), missing stage time, queue > cycle clamp
- Invalid inputs raise informative errors
- Immutability: input DataFrame is not mutated
- Determinism: repeated calls yield identical results
- Sorting: worst offender appears first
- Round-trip via to_dataframe()
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.queue_time_analyzer import (
    HIGH_SEVERITY_MAX,
    LOW_SEVERITY_MAX,
    MODERATE_SEVERITY_MAX,
    QueueTimeReport,
    TruckQueueStats,
    analyze_queue_time,
    classify_queue_severity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fleet_df() -> pd.DataFrame:
    """Three-truck fleet with mixed queue severities."""
    return pd.DataFrame(
        {
            "truck_id": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "queue_time_min": [2.0, 3.0, 8.0, 10.0, 14.0, 16.0],
            "total_cycle_min": [50.0, 60.0, 50.0, 50.0, 50.0, 50.0],
            "pit_name": ["North"] * 6,
        }
    )


# ---------------------------------------------------------------------------
# classify_queue_severity
# ---------------------------------------------------------------------------


class TestClassifyQueueSeverity:
    def test_low_at_zero(self):
        assert classify_queue_severity(0.0) == "low"

    def test_low_at_boundary(self):
        assert classify_queue_severity(LOW_SEVERITY_MAX) == "low"

    def test_moderate(self):
        assert classify_queue_severity(0.10) == "moderate"

    def test_moderate_at_boundary(self):
        assert classify_queue_severity(MODERATE_SEVERITY_MAX) == "moderate"

    def test_high(self):
        assert classify_queue_severity(0.20) == "high"

    def test_high_at_boundary(self):
        assert classify_queue_severity(HIGH_SEVERITY_MAX) == "high"

    def test_critical(self):
        assert classify_queue_severity(0.30) == "critical"

    def test_critical_at_one(self):
        assert classify_queue_severity(1.0) == "critical"

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="queue_ratio must be in"):
            classify_queue_severity(-0.01)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="queue_ratio must be in"):
            classify_queue_severity(1.01)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must not be NaN"):
            classify_queue_severity(float("nan"))


# ---------------------------------------------------------------------------
# analyze_queue_time -- happy path
# ---------------------------------------------------------------------------


class TestAnalyzeQueueTimeHappyPath:
    def test_returns_report(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        assert isinstance(report, QueueTimeReport)

    def test_three_truck_stats(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        assert len(report.truck_stats) == 3

    def test_worst_truck_first(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        assert report.truck_stats[0].truck_id == "T3"
        assert report.worst_truck == "T3"

    def test_fleet_queue_ratio(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        # total queue 53, total cycle 310 -> 0.171...
        assert report.fleet_queue_ratio == pytest.approx(53.0 / 310.0, rel=1e-3)

    def test_t1_low_severity(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        t1 = next(s for s in report.truck_stats if s.truck_id == "T1")
        # 5 / 110 = 0.0454 -> low
        assert t1.severity == "low"

    def test_t2_severity(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        t2 = next(s for s in report.truck_stats if s.truck_id == "T2")
        # 18 / 100 = 0.18 -> high
        assert t2.severity == "high"

    def test_t3_critical(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        t3 = next(s for s in report.truck_stats if s.truck_id == "T3")
        # 30 / 100 = 0.30 -> critical
        assert t3.severity == "critical"
        assert report.n_critical_trucks == 1

    def test_avg_queue_min_t1(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        t1 = next(s for s in report.truck_stats if s.truck_id == "T1")
        assert t1.avg_queue_min == pytest.approx(2.5)
        assert t1.n_cycles == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self):
        report = analyze_queue_time(pd.DataFrame())
        assert report.truck_stats == ()
        assert math.isnan(report.fleet_queue_ratio)
        assert report.worst_truck is None
        assert report.n_critical_trucks == 0

    def test_missing_truck_col(self):
        df = pd.DataFrame(
            {"queue_time_min": [1.0], "total_cycle_min": [50.0]}
        )
        report = analyze_queue_time(df)
        assert report.truck_stats == ()

    def test_missing_queue_col(self):
        df = pd.DataFrame(
            {"truck_id": ["T1"], "total_cycle_min": [50.0]}
        )
        report = analyze_queue_time(df)
        assert report.truck_stats == ()

    def test_missing_cycle_col(self):
        df = pd.DataFrame(
            {"truck_id": ["T1"], "queue_time_min": [3.0]}
        )
        report = analyze_queue_time(df)
        assert report.truck_stats == ()

    def test_zero_cycles_skipped(self):
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T1"],
                "queue_time_min": [3.0, 4.0],
                "total_cycle_min": [0.0, 0.0],
            }
        )
        report = analyze_queue_time(df)
        assert report.truck_stats == ()
        assert math.isnan(report.fleet_queue_ratio)

    def test_negative_queue_skipped(self):
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T1"],
                "queue_time_min": [-2.0, 5.0],
                "total_cycle_min": [50.0, 50.0],
            }
        )
        report = analyze_queue_time(df)
        # only the one valid row remains
        assert report.truck_stats[0].n_cycles == 1
        assert report.truck_stats[0].total_queue_min == pytest.approx(5.0)

    def test_single_truck(self):
        df = pd.DataFrame(
            {
                "truck_id": ["TRK01"],
                "queue_time_min": [5.0],
                "total_cycle_min": [50.0],
            }
        )
        report = analyze_queue_time(df)
        assert len(report.truck_stats) == 1
        assert report.worst_truck == "TRK01"
        assert report.fleet_queue_ratio == pytest.approx(0.1)

    def test_all_same_route_single_pit(self):
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T2", "T3", "T4"],
                "queue_time_min": [4.0, 5.0, 6.0, 4.5],
                "total_cycle_min": [50.0] * 4,
                "pit_name": ["North"] * 4,
                "route": ["North->Crusher"] * 4,
            }
        )
        report = analyze_queue_time(df)
        assert len(report.truck_stats) == 4
        # All low severity (queue ratios 0.08-0.12 -> moderate)
        assert all(s.severity == "moderate" for s in report.truck_stats)

    def test_missing_stage_time_uses_fallback(self):
        # No total_cycle_min -- falls back to computed_cycle_min
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "queue_time_min": [5.0],
                "computed_cycle_min": [50.0],
            }
        )
        report = analyze_queue_time(df)
        assert len(report.truck_stats) == 1
        assert report.truck_stats[0].queue_ratio == pytest.approx(0.1)

    def test_explicit_cycle_time_col(self):
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "queue_time_min": [5.0],
                "my_custom_cycle": [50.0],
            }
        )
        report = analyze_queue_time(df, cycle_time_col="my_custom_cycle")
        assert report.truck_stats[0].queue_ratio == pytest.approx(0.1)

    def test_queue_greater_than_cycle_is_clamped(self):
        df = pd.DataFrame(
            {
                "truck_id": ["T1"],
                "queue_time_min": [80.0],
                "total_cycle_min": [50.0],
            }
        )
        report = analyze_queue_time(df)
        # Clamped to cycle -> ratio == 1.0 -> critical
        assert report.truck_stats[0].queue_ratio == pytest.approx(1.0)
        assert report.truck_stats[0].severity == "critical"

    def test_nan_values_skipped(self):
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T1"],
                "queue_time_min": [float("nan"), 4.0],
                "total_cycle_min": [50.0, 50.0],
            }
        )
        report = analyze_queue_time(df)
        assert report.truck_stats[0].n_cycles == 1


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_input_not_mutated(self, fleet_df):
        snapshot = fleet_df.copy(deep=True)
        analyze_queue_time(fleet_df)
        pd.testing.assert_frame_equal(fleet_df, snapshot)

    def test_deterministic(self, fleet_df):
        a = analyze_queue_time(fleet_df)
        b = analyze_queue_time(fleet_df)
        assert a == b

    def test_report_is_immutable(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        with pytest.raises(Exception):
            report.worst_truck = "X"  # frozen dataclass

    def test_truck_stat_is_immutable(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        with pytest.raises(Exception):
            report.truck_stats[0].severity = "low"

    def test_to_dataframe_round_trip(self, fleet_df):
        report = analyze_queue_time(fleet_df)
        df = report.to_dataframe()
        assert list(df.columns) == [
            "truck_id",
            "n_cycles",
            "total_queue_min",
            "total_cycle_min",
            "avg_queue_min",
            "queue_ratio",
            "severity",
        ]
        assert len(df) == 3
        # First row is the worst offender
        assert df.iloc[0]["truck_id"] == "T3"

    def test_to_dataframe_empty(self):
        report = analyze_queue_time(pd.DataFrame())
        df = report.to_dataframe()
        assert df.empty
        assert "queue_ratio" in df.columns


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            analyze_queue_time([1, 2, 3])  # type: ignore[arg-type]

    def test_non_dataframe_dict_raises(self):
        with pytest.raises(TypeError):
            analyze_queue_time({"truck_id": ["T1"]})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration with sample data
# ---------------------------------------------------------------------------


class TestSampleDataIntegration:
    def test_runs_on_sample_csv(self):
        df = pd.read_csv("demo/sample_data.csv")
        report = analyze_queue_time(df)
        # Sample has 5 trucks
        assert len(report.truck_stats) == 5
        assert 0.0 < report.fleet_queue_ratio < 1.0
        assert report.worst_truck is not None
