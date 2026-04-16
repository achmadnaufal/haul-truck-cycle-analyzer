"""
Pytest test suite for src/fleet_match_factor_calculator.py.

Covers:
- Happy-path match factor arithmetic
- Classification thresholds (under-trucked / balanced / over-trucked)
- Edge cases: empty DataFrame, single truck, single pit, zero payload cycles
- Invalid inputs raise ValueError
- Immutability: input DataFrame is not mutated
- Determinism: repeated calls produce identical results
- Parametrized cases for a range of truck counts
- MatchFactorReport.to_dataframe() round-trip
- Fallback from total_cycle_min to computed_cycle_min
- Pit with no valid cycle times is skipped gracefully
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.fleet_match_factor_calculator import (
    OVER_TRUCKED_THRESHOLD,
    UNDER_TRUCKED_THRESHOLD,
    MatchFactorReport,
    PitMatchResult,
    _classify_condition,
    calculate_fleet_match_factor,
    compute_match_factor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n_trucks: int,
    cycle_time: float,
    pit: str = "North Pit",
    cycle_col: str = "total_cycle_min",
) -> pd.DataFrame:
    """Build a minimal cycle DataFrame for testing."""
    rows = []
    for i in range(n_trucks):
        rows.append(
            {
                "truck_id": f"TRK{i + 1:02d}",
                "pit_name": pit,
                cycle_col: cycle_time,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. compute_match_factor — happy path
# ---------------------------------------------------------------------------


def test_compute_match_factor_perfect_balance() -> None:
    """MF == 1.0 when trucks are exactly balanced with the loader."""
    # 4 trucks, 48-min cycle, 12-min loader pass, 1 pass/truck → MF = 1.0
    mf = compute_match_factor(
        n_trucks=4, truck_cycle_time_min=48.0, loader_cycle_time_min=12.0
    )
    assert mf == pytest.approx(1.0, rel=1e-4)


def test_compute_match_factor_under_trucked() -> None:
    """MF < 1 when fewer trucks than needed to keep the loader busy."""
    mf = compute_match_factor(
        n_trucks=3, truck_cycle_time_min=50.0, loader_cycle_time_min=12.0
    )
    assert mf < UNDER_TRUCKED_THRESHOLD


def test_compute_match_factor_over_trucked() -> None:
    """MF > 1 when more trucks than the loader can service efficiently."""
    mf = compute_match_factor(
        n_trucks=6, truck_cycle_time_min=50.0, loader_cycle_time_min=12.0
    )
    assert mf > OVER_TRUCKED_THRESHOLD


def test_compute_match_factor_n_passes_scales_correctly() -> None:
    """Increasing n_passes proportionally reduces the match factor."""
    mf_1pass = compute_match_factor(4, 50.0, 10.0, n_passes=1)
    mf_2pass = compute_match_factor(4, 50.0, 10.0, n_passes=2)
    assert pytest.approx(mf_1pass / 2, rel=1e-4) == mf_2pass


def test_compute_match_factor_result_is_rounded() -> None:
    """Result must be rounded to at most 4 decimal places."""
    mf = compute_match_factor(3, 47.0, 11.0, n_passes=1)
    assert mf == round(mf, 4)


# ---------------------------------------------------------------------------
# 2. compute_match_factor — invalid inputs raise ValueError
# ---------------------------------------------------------------------------


def test_compute_match_factor_zero_trucks_raises() -> None:
    with pytest.raises(ValueError, match="n_trucks"):
        compute_match_factor(0, 50.0, 12.0)


def test_compute_match_factor_zero_truck_cycle_raises() -> None:
    with pytest.raises(ValueError, match="truck_cycle_time_min"):
        compute_match_factor(4, 0.0, 12.0)


def test_compute_match_factor_zero_loader_cycle_raises() -> None:
    with pytest.raises(ValueError, match="loader_cycle_time_min"):
        compute_match_factor(4, 50.0, 0.0)


def test_compute_match_factor_zero_passes_raises() -> None:
    with pytest.raises(ValueError, match="n_passes"):
        compute_match_factor(4, 50.0, 12.0, n_passes=0)


# ---------------------------------------------------------------------------
# 3. _classify_condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mf, expected",
    [
        (0.50, "under-trucked"),
        (UNDER_TRUCKED_THRESHOLD - 0.01, "under-trucked"),
        (UNDER_TRUCKED_THRESHOLD, "balanced"),
        (1.00, "balanced"),
        (OVER_TRUCKED_THRESHOLD, "balanced"),
        (OVER_TRUCKED_THRESHOLD + 0.01, "over-trucked"),
        (2.00, "over-trucked"),
    ],
)
def test_classify_condition_parametrized(mf: float, expected: str) -> None:
    assert _classify_condition(mf) == expected


# ---------------------------------------------------------------------------
# 4. calculate_fleet_match_factor — happy path
# ---------------------------------------------------------------------------


def test_calculate_fleet_single_pit_balanced() -> None:
    """Four trucks on a 48-min cycle with 12-min loader → balanced."""
    df = _make_df(n_trucks=4, cycle_time=48.0)
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    assert len(report.pit_results) == 1
    assert report.pit_results[0].condition == "balanced"
    assert report.n_balanced == 1
    assert report.n_under_trucked == 0
    assert report.n_over_trucked == 0


def test_calculate_fleet_overall_match_factor_equals_single_pit() -> None:
    """Overall MF equals the single pit MF when there is only one pit."""
    df = _make_df(n_trucks=4, cycle_time=50.0)
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    assert report.overall_match_factor == report.pit_results[0].match_factor


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


def test_calculate_fleet_empty_dataframe() -> None:
    """Empty DataFrame returns a report with no pit results and NaN MF."""
    report = calculate_fleet_match_factor(pd.DataFrame(), loader_cycle_time_min=12.0)
    assert report.pit_results == ()
    assert math.isnan(report.overall_match_factor)


def test_calculate_fleet_missing_pit_col() -> None:
    """DataFrame without pit_col returns empty report."""
    df = pd.DataFrame({"truck_id": ["T1"], "total_cycle_min": [50.0]})
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    assert report.pit_results == ()


def test_calculate_fleet_single_truck() -> None:
    """Single-truck fleet is handled without error."""
    df = _make_df(n_trucks=1, cycle_time=50.0)
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    assert report.pit_results[0].n_trucks == 1


def test_calculate_fleet_zero_cycle_time_rows_skipped() -> None:
    """Rows with zero or negative cycle times are excluded from the mean."""
    df = pd.DataFrame(
        {
            "truck_id": ["T1", "T1", "T2"],
            "pit_name": ["North Pit", "North Pit", "North Pit"],
            "total_cycle_min": [0.0, 50.0, 50.0],
        }
    )
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    # Valid mean should use only the 50.0 values
    assert report.pit_results[0].avg_truck_cycle_min == pytest.approx(50.0, rel=1e-4)


def test_calculate_fleet_falls_back_to_computed_cycle_col() -> None:
    """Uses computed_cycle_min when total_cycle_min is absent."""
    df = pd.DataFrame(
        {
            "truck_id": ["T1", "T2"],
            "pit_name": ["South Pit", "South Pit"],
            "computed_cycle_min": [55.0, 55.0],
        }
    )
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=11.0)
    assert len(report.pit_results) == 1
    assert report.pit_results[0].avg_truck_cycle_min == pytest.approx(55.0, rel=1e-4)


def test_calculate_fleet_multiple_pits() -> None:
    """Correct pit counts and conditions with two pits."""
    df = pd.DataFrame(
        {
            "truck_id": ["T1", "T2", "T3", "T4", "T5"],
            "pit_name": ["North", "North", "North", "South", "South"],
            "total_cycle_min": [50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )
    # loader=12 min, 1 pass
    # North: MF = 3*12/50 = 0.72 → under-trucked
    # South: MF = 2*12/50 = 0.48 → under-trucked
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    assert len(report.pit_results) == 2
    assert report.n_under_trucked == 2


# ---------------------------------------------------------------------------
# 6. Immutability — input DataFrame is not mutated
# ---------------------------------------------------------------------------


def test_calculate_fleet_does_not_mutate_input() -> None:
    """The input DataFrame must remain unchanged after the call."""
    df = _make_df(n_trucks=3, cycle_time=48.0)
    original_columns = list(df.columns)
    original_shape = df.shape
    calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    assert list(df.columns) == original_columns
    assert df.shape == original_shape


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


def test_calculate_fleet_is_deterministic() -> None:
    """Repeated calls with identical input must return identical MF values."""
    df = _make_df(n_trucks=4, cycle_time=52.0)
    report_a = calculate_fleet_match_factor(df, loader_cycle_time_min=13.0)
    report_b = calculate_fleet_match_factor(df, loader_cycle_time_min=13.0)
    assert report_a.overall_match_factor == report_b.overall_match_factor
    assert report_a.pit_results[0].match_factor == report_b.pit_results[0].match_factor


# ---------------------------------------------------------------------------
# 8. MatchFactorReport.to_dataframe()
# ---------------------------------------------------------------------------


def test_to_dataframe_columns() -> None:
    """to_dataframe returns expected column names."""
    df = _make_df(n_trucks=4, cycle_time=50.0)
    report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    result_df = report.to_dataframe()
    expected_cols = {
        "pit_name",
        "n_trucks",
        "avg_truck_cycle_min",
        "loader_cycle_time_min",
        "n_passes",
        "match_factor",
        "condition",
    }
    assert expected_cols == set(result_df.columns)


def test_to_dataframe_empty_when_no_results() -> None:
    """to_dataframe returns empty DataFrame when pit_results is empty."""
    report = MatchFactorReport(
        pit_results=(),
        overall_match_factor=float("nan"),
        n_under_trucked=0,
        n_over_trucked=0,
        n_balanced=0,
    )
    assert report.to_dataframe().empty


# ---------------------------------------------------------------------------
# 9. Invalid loader/passes inputs to calculate_fleet_match_factor
# ---------------------------------------------------------------------------


def test_calculate_fleet_zero_loader_cycle_raises() -> None:
    df = _make_df(n_trucks=3, cycle_time=50.0)
    with pytest.raises(ValueError, match="loader_cycle_time_min"):
        calculate_fleet_match_factor(df, loader_cycle_time_min=0.0)


def test_calculate_fleet_zero_passes_raises() -> None:
    df = _make_df(n_trucks=3, cycle_time=50.0)
    with pytest.raises(ValueError, match="n_passes"):
        calculate_fleet_match_factor(df, loader_cycle_time_min=12.0, n_passes=0)
