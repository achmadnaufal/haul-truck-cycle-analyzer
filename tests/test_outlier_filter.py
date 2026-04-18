"""Tests for :mod:`src.outlier_filter`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.outlier_filter import (
    DEFAULT_IQR_MULTIPLIER,
    OutlierReport,
    filter_outliers,
    flag_outliers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_cycles() -> pd.DataFrame:
    """Ten realistic cycle rows with a single obvious outlier (row index 9)."""
    return pd.DataFrame(
        {
            "truck_id": [f"TRK{i:02d}" for i in range(1, 11)],
            "total_cycle_min": [
                50.0, 51.2, 49.8, 50.5, 52.0, 48.7, 51.5, 50.9, 49.5, 500.0,
            ],
            "payload_tonnes": [
                220.0, 218.5, 221.0, 219.5, 222.0, 220.5, 217.0, 223.0, 219.0, 220.0,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_filter_outliers_iqr_removes_large_outlier(normal_cycles):
    filtered, report = filter_outliers(normal_cycles, method="iqr")
    assert report.n_total == 10
    assert report.n_outliers == 1
    assert report.n_kept == 9
    # The 500-minute cycle must be dropped.
    assert 500.0 not in filtered["total_cycle_min"].values
    assert isinstance(report, OutlierReport)
    assert report.method == "iqr"
    assert report.outlier_ratio == pytest.approx(0.1)


def test_filter_outliers_zscore_matches_iqr_on_clear_outlier(normal_cycles):
    filtered, report = filter_outliers(
        normal_cycles, method="zscore", zscore_threshold=2.0
    )
    # At z=2 the 500-minute cycle is still far outside.
    assert 500.0 not in filtered["total_cycle_min"].values
    assert report.method == "zscore"
    assert report.n_outliers >= 1


def test_flag_outliers_preserves_row_count_and_marks_outlier(normal_cycles):
    flagged = flag_outliers(normal_cycles, method="iqr")
    assert len(flagged) == len(normal_cycles)
    # Last row (500 min) must be flagged, first row must not.
    assert flagged["is_outlier"].iloc[-1] is np.True_ or flagged["is_outlier"].iloc[-1]
    assert not flagged["is_outlier"].iloc[0]
    # Default column name present.
    assert "is_outlier" in flagged.columns


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_filter_outliers_empty_dataframe_returns_empty_report():
    empty = pd.DataFrame({"total_cycle_min": pd.Series([], dtype=float)})
    filtered, report = filter_outliers(empty)
    assert filtered.empty
    assert report.n_total == 0
    assert report.n_kept == 0
    assert report.outlier_ratio == 0.0


def test_filter_outliers_constant_column_flags_nothing():
    df = pd.DataFrame({"total_cycle_min": [50.0] * 5})
    filtered, report = filter_outliers(df, method="zscore")
    assert report.n_outliers == 0
    assert len(filtered) == 5


def test_filter_outliers_nan_rows_are_dropped():
    df = pd.DataFrame({"total_cycle_min": [50.0, 51.0, np.nan, 49.0, 50.5]})
    filtered, report = filter_outliers(df, method="iqr")
    assert report.n_outliers == 1
    assert filtered["total_cycle_min"].notna().all()


def test_filter_outliers_input_not_mutated(normal_cycles):
    original = normal_cycles.copy(deep=True)
    _ = filter_outliers(normal_cycles, method="iqr")
    pd.testing.assert_frame_equal(normal_cycles, original)


def test_filter_outliers_rejects_unknown_method(normal_cycles):
    with pytest.raises(ValueError, match="method"):
        filter_outliers(normal_cycles, method="bogus")


def test_filter_outliers_rejects_missing_column(normal_cycles):
    with pytest.raises(ValueError, match="not found"):
        filter_outliers(normal_cycles, column="no_such_col")


def test_filter_outliers_rejects_non_numeric_column():
    df = pd.DataFrame({"total_cycle_min": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="numeric"):
        filter_outliers(df)


def test_filter_outliers_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        filter_outliers([1, 2, 3])  # type: ignore[arg-type]


def test_filter_outliers_rejects_negative_iqr_multiplier(normal_cycles):
    with pytest.raises(ValueError, match="iqr_multiplier"):
        filter_outliers(normal_cycles, iqr_multiplier=-1.0)


def test_filter_outliers_rejects_non_positive_zscore_threshold(normal_cycles):
    with pytest.raises(ValueError, match="zscore_threshold"):
        filter_outliers(normal_cycles, method="zscore", zscore_threshold=0.0)


def test_report_is_frozen():
    report = OutlierReport(
        column="total_cycle_min",
        method="iqr",
        lower_bound=0.0,
        upper_bound=100.0,
        n_total=5,
        n_outliers=1,
        n_kept=4,
    )
    with pytest.raises(Exception):
        report.n_total = 99  # type: ignore[misc]


def test_default_iqr_multiplier_is_classic_tukey():
    assert DEFAULT_IQR_MULTIPLIER == 1.5


def test_flag_outliers_custom_flag_column(normal_cycles):
    flagged = flag_outliers(normal_cycles, flag_column="abnormal")
    assert "abnormal" in flagged.columns
    assert flagged["abnormal"].dtype == bool
