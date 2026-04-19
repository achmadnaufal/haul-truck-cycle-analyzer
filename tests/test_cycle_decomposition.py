"""Unit tests for :mod:`src.cycle_decomposition`.

Test classes:
    TestDecomposeHappyPath      - Basic decomposition on clean inputs.
    TestDecomposeMatchFactor    - Match factor math via compute_match_factor.
    TestDecomposeBottleneck     - Fleet bottleneck identification.
    TestDecomposeEdgeCases      - Empty, NaN, zero-cycle, missing stages.
    TestRankByMedian            - Median-ordered stage ranking.
    TestImmutability            - Immutable result objects and inputs.
    TestDataFrameRoundTrip      - to_dataframe round-trip.
    TestSampleDataIntegration   - End-to-end on demo/sample_data.csv.

Run with:
    pytest tests/test_cycle_decomposition.py -v
"""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup -- allow imports without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cycle_decomposition import (
    STAGE_ORDER,
    CycleDecompositionReport,
    StageStats,
    decompose_cycle,
    rank_stages_by_median,
)
from src.fleet_match_factor_calculator import compute_match_factor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def balanced_df() -> pd.DataFrame:
    """Small fleet with hauling clearly dominant -- the realistic case."""
    return pd.DataFrame(
        {
            "truck_id": ["T1", "T1", "T2", "T2", "T3"],
            "loading_time_min":  [8.0, 9.0, 8.5, 7.5, 8.2],
            "hauling_time_min":  [20.0, 22.0, 21.0, 19.0, 20.5],
            "dumping_time_min":  [3.0, 3.5, 2.8, 2.6, 3.1],
            "return_time_min":   [17.0, 18.0, 16.5, 15.8, 16.9],
            "queue_time_min":    [4.0, 5.0, 3.5, 3.0, 4.2],
        }
    )


@pytest.fixture()
def queue_dominant_df() -> pd.DataFrame:
    """Fleet where queue time is the largest stage -- over-trucked signal."""
    return pd.DataFrame(
        {
            "truck_id": ["T1", "T2", "T3"],
            "loading_time_min":  [5.0, 5.0, 5.0],
            "hauling_time_min":  [10.0, 10.0, 10.0],
            "dumping_time_min":  [2.0, 2.0, 2.0],
            "return_time_min":   [9.0, 9.0, 9.0],
            "queue_time_min":    [25.0, 26.0, 24.0],
        }
    )


@pytest.fixture()
def alias_df() -> pd.DataFrame:
    """Timestamp-schema aliases (load_time_min etc.)."""
    return pd.DataFrame(
        {
            "truck_id": ["T1", "T2"],
            "load_time_min":  [8.0, 9.0],
            "haul_time_min":  [20.0, 22.0],
            "dump_time_min":  [3.0, 3.5],
            "return_time_min":  [17.0, 18.0],
        }
    )


# ---------------------------------------------------------------------------
# Test 1 -- Happy path decomposition
# ---------------------------------------------------------------------------


class TestDecomposeHappyPath:
    """Decomposition on clean, well-populated inputs."""

    def test_returns_report_instance(self, balanced_df: pd.DataFrame) -> None:
        """Function must return a CycleDecompositionReport."""
        report = decompose_cycle(balanced_df)
        assert isinstance(report, CycleDecompositionReport)

    def test_all_five_stages_populated(self, balanced_df: pd.DataFrame) -> None:
        """When all five stage columns are present, all five must be in stats."""
        report = decompose_cycle(balanced_df)
        assert set(report.stage_stats.keys()) == set(STAGE_ORDER)

    def test_load_stage_mean_correct(self, balanced_df: pd.DataFrame) -> None:
        """Load stage mean must match direct pandas mean of the column."""
        report = decompose_cycle(balanced_df)
        expected = float(balanced_df["loading_time_min"].mean())
        assert report.stage_stats["load"].mean_min == pytest.approx(
            expected, rel=1e-4
        )

    def test_haul_stage_median_correct(self, balanced_df: pd.DataFrame) -> None:
        """Haul stage median must match direct pandas median of the column."""
        report = decompose_cycle(balanced_df)
        expected = float(balanced_df["hauling_time_min"].median())
        assert report.stage_stats["haul"].median_min == pytest.approx(
            expected, rel=1e-4
        )

    def test_shares_sum_to_one(self, balanced_df: pd.DataFrame) -> None:
        """Share of cycle across all stages must sum to exactly 1.0."""
        report = decompose_cycle(balanced_df)
        total_share = sum(s.share_of_cycle for s in report.stage_stats.values())
        assert total_share == pytest.approx(1.0, rel=1e-3)

    def test_mean_total_cycle_matches_row_sums(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """mean_total_cycle_min must equal the mean of per-row stage sums."""
        report = decompose_cycle(balanced_df)
        per_row_total = (
            balanced_df[
                [
                    "loading_time_min",
                    "hauling_time_min",
                    "dumping_time_min",
                    "return_time_min",
                    "queue_time_min",
                ]
            ]
            .sum(axis=1)
            .mean()
        )
        assert report.mean_total_cycle_min == pytest.approx(
            per_row_total, rel=1e-3
        )

    def test_n_cycles_count(self, balanced_df: pd.DataFrame) -> None:
        """n_cycles for every stage must equal the number of input rows."""
        report = decompose_cycle(balanced_df)
        for stage_stats in report.stage_stats.values():
            assert stage_stats.n_cycles == len(balanced_df)


# ---------------------------------------------------------------------------
# Test 2 -- Match factor sanity (M = 1 for balanced setup)
# ---------------------------------------------------------------------------


class TestMatchFactorMath:
    """Verify match-factor formula on canonical inputs."""

    def test_balanced_setup_returns_one(self) -> None:
        """M = (n_trucks * loader_cycle) / (truck_cycle * n_passes); balanced = 1."""
        # 4 trucks, 50 min cycle, loader 12.5 min per pass, 1 pass -> exact 1.0
        mf = compute_match_factor(
            n_trucks=4,
            truck_cycle_time_min=50.0,
            loader_cycle_time_min=12.5,
            n_passes=1,
        )
        assert mf == pytest.approx(1.0, rel=1e-6)

    def test_under_trucked_setup_less_than_one(self) -> None:
        """Fewer trucks than balanced -> MF < 1 (under-trucked, loader bottleneck)."""
        mf = compute_match_factor(
            n_trucks=3,
            truck_cycle_time_min=50.0,
            loader_cycle_time_min=12.5,
            n_passes=1,
        )
        assert mf < 1.0

    def test_over_trucked_setup_greater_than_one(self) -> None:
        """More trucks than balanced -> MF > 1 (over-trucked, truck bottleneck)."""
        mf = compute_match_factor(
            n_trucks=5,
            truck_cycle_time_min=50.0,
            loader_cycle_time_min=12.5,
            n_passes=1,
        )
        assert mf > 1.0

    def test_multi_pass_shovel_halves_mf(self) -> None:
        """Doubling n_passes halves the match factor (formula sanity)."""
        mf_1pass = compute_match_factor(
            n_trucks=4,
            truck_cycle_time_min=50.0,
            loader_cycle_time_min=12.5,
            n_passes=1,
        )
        mf_2pass = compute_match_factor(
            n_trucks=4,
            truck_cycle_time_min=50.0,
            loader_cycle_time_min=12.5,
            n_passes=2,
        )
        assert mf_2pass == pytest.approx(mf_1pass / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 3 -- Bottleneck identification (the new feature's core output)
# ---------------------------------------------------------------------------


class TestDecomposeBottleneck:
    """Fleet-level bottleneck stage detection."""

    def test_haul_is_bottleneck_on_balanced_fleet(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """When hauling dominates, bottleneck_stage must be 'haul'."""
        report = decompose_cycle(balanced_df)
        assert report.bottleneck_stage == "haul"

    def test_queue_is_bottleneck_on_over_trucked_fleet(
        self, queue_dominant_df: pd.DataFrame
    ) -> None:
        """When queue dominates, bottleneck_stage must be 'queue'."""
        report = decompose_cycle(queue_dominant_df)
        assert report.bottleneck_stage == "queue"

    def test_bottleneck_share_matches_stage_share(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """bottleneck_share must equal the dominant stage's share_of_cycle."""
        report = decompose_cycle(balanced_df)
        dominant = report.stage_stats[report.bottleneck_stage]
        assert report.bottleneck_share == pytest.approx(
            dominant.share_of_cycle, rel=1e-6
        )

    def test_bottleneck_share_in_valid_range(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """bottleneck_share must be in (0, 1] for any non-degenerate fleet."""
        report = decompose_cycle(balanced_df)
        assert 0.0 < report.bottleneck_share <= 1.0

    def test_loading_bottleneck_single_stage(self) -> None:
        """When only loading is present, loading must be the bottleneck."""
        df = pd.DataFrame({"loading_time_min": [10.0, 12.0, 11.0]})
        report = decompose_cycle(df)
        assert report.bottleneck_stage == "load"
        assert report.bottleneck_share == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 4 -- Edge cases (the prompt's required checklist)
# ---------------------------------------------------------------------------


class TestDecomposeEdgeCases:
    """Zero cycles, negatives, missing stages, NaN, duplicates, empty DF."""

    def test_empty_dataframe_returns_empty_report(self) -> None:
        """Empty DataFrame must return an empty report, not raise."""
        report = decompose_cycle(pd.DataFrame())
        assert report.stage_stats == {}
        assert report.total_cycles == 0
        assert report.mean_total_cycle_min == 0.0
        assert report.bottleneck_stage is None
        assert report.bottleneck_share == 0.0

    def test_all_nan_rows_dropped_as_empty(self) -> None:
        """A DataFrame of only NaN rows behaves like empty input."""
        df = pd.DataFrame(
            {
                "loading_time_min": [np.nan, np.nan],
                "hauling_time_min": [np.nan, np.nan],
            }
        )
        report = decompose_cycle(df)
        assert report.stage_stats == {}
        assert report.bottleneck_stage is None

    def test_dataframe_without_stage_columns_returns_empty(self) -> None:
        """DataFrame with no recognised stage columns -> empty report."""
        df = pd.DataFrame({"truck_id": ["T1", "T2"], "foo": [1, 2]})
        report = decompose_cycle(df)
        assert report.stage_stats == {}
        assert report.bottleneck_stage is None
        # total_cycles still reflects the non-empty input.
        assert report.total_cycles == 2

    def test_negative_durations_excluded(self) -> None:
        """Negative stage durations are filtered out, not clamped."""
        df = pd.DataFrame(
            {
                "loading_time_min": [8.0, -1.0, 10.0],
                "hauling_time_min": [20.0, 22.0, 21.0],
            }
        )
        report = decompose_cycle(df)
        # Load stage should only count the two positive rows.
        assert report.stage_stats["load"].n_cycles == 2
        assert report.stage_stats["load"].mean_min == pytest.approx(9.0, rel=1e-4)

    def test_nan_cells_excluded_per_stage(self) -> None:
        """NaN values in a stage column are dropped for that stage only."""
        df = pd.DataFrame(
            {
                "loading_time_min": [8.0, np.nan, 10.0],
                "hauling_time_min": [20.0, 22.0, 21.0],
            }
        )
        report = decompose_cycle(df)
        assert report.stage_stats["load"].n_cycles == 2
        assert report.stage_stats["haul"].n_cycles == 3

    def test_zero_cycle_all_zero_stages(self) -> None:
        """Zero-valued stages are valid; the bottleneck falls back to zero mean."""
        df = pd.DataFrame(
            {
                "loading_time_min": [0.0, 0.0],
                "hauling_time_min": [0.0, 0.0],
            }
        )
        report = decompose_cycle(df)
        # Every stage has mean 0; total cycle is 0; share falls back to 0.
        assert report.mean_total_cycle_min == 0.0
        for stats in report.stage_stats.values():
            assert stats.share_of_cycle == 0.0

    def test_missing_stage_column_silently_skipped(self) -> None:
        """A DataFrame missing some stages must still produce stats for the rest."""
        df = pd.DataFrame(
            {
                "loading_time_min": [8.0, 9.0],
                "hauling_time_min": [20.0, 22.0],
            }
        )
        report = decompose_cycle(df)
        # queue/dump/return absent -> only load and haul reported.
        assert set(report.stage_stats.keys()) == {"load", "haul"}

    def test_duplicate_truck_id_rows_all_counted(self) -> None:
        """Duplicate truck_id is not a validity issue for decomposition."""
        df = pd.DataFrame(
            {
                "truck_id": ["T1", "T1", "T1"],
                "loading_time_min": [8.0, 9.0, 10.0],
                "hauling_time_min": [20.0, 22.0, 21.0],
            }
        )
        report = decompose_cycle(df)
        assert report.stage_stats["load"].n_cycles == 3

    def test_single_row_dataframe(self) -> None:
        """A single-row DataFrame must produce stats with n_cycles=1."""
        df = pd.DataFrame(
            {
                "loading_time_min": [8.0],
                "hauling_time_min": [20.0],
            }
        )
        report = decompose_cycle(df)
        assert report.stage_stats["load"].n_cycles == 1
        assert report.stage_stats["load"].mean_min == pytest.approx(8.0, rel=1e-4)
        # Sample std with one sample is 0.0 by convention.
        assert report.stage_stats["load"].std_min == 0.0

    def test_non_dataframe_input_raises(self) -> None:
        """Passing a non-DataFrame must raise TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame"):
            decompose_cycle([1, 2, 3])  # type: ignore[arg-type]

    def test_alias_columns_accepted(self, alias_df: pd.DataFrame) -> None:
        """Timestamp-schema aliases (load_time_min etc.) must resolve correctly."""
        report = decompose_cycle(alias_df)
        assert "load" in report.stage_stats
        assert report.stage_stats["load"].column == "load_time_min"
        assert report.stage_stats["haul"].column == "haul_time_min"


# ---------------------------------------------------------------------------
# Test 5 -- Median-based ranking helper
# ---------------------------------------------------------------------------


class TestRankByMedian:
    """Stage ranking by median duration."""

    def test_rank_matches_expected_order(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """Ranked stages must be ordered largest-median first."""
        ranking = rank_stages_by_median(balanced_df)
        medians = [m for _, m in ranking]
        assert medians == sorted(medians, reverse=True)

    def test_rank_first_matches_bottleneck_on_balanced_fleet(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """On a well-behaved fleet, median ranking agrees with mean ranking."""
        ranking = rank_stages_by_median(balanced_df)
        assert ranking[0][0] == "haul"

    def test_rank_empty_on_empty_df(self) -> None:
        """Empty DataFrame must yield an empty ranking tuple."""
        assert rank_stages_by_median(pd.DataFrame()) == ()

    def test_rank_contains_only_present_stages(self) -> None:
        """Rank tuple length must equal number of stages in the input."""
        df = pd.DataFrame(
            {
                "loading_time_min": [8.0, 9.0],
                "hauling_time_min": [20.0, 22.0],
            }
        )
        ranking = rank_stages_by_median(df)
        assert len(ranking) == 2


# ---------------------------------------------------------------------------
# Test 6 -- Immutability guarantees
# ---------------------------------------------------------------------------


class TestImmutability:
    """Frozen dataclass and non-mutation guarantees."""

    def test_stage_stats_is_frozen(self, balanced_df: pd.DataFrame) -> None:
        """StageStats must be a frozen dataclass (cannot reassign fields)."""
        report = decompose_cycle(balanced_df)
        stats = report.stage_stats["load"]
        with pytest.raises(FrozenInstanceError):
            stats.mean_min = 999.9  # type: ignore[misc]

    def test_report_is_frozen(self, balanced_df: pd.DataFrame) -> None:
        """CycleDecompositionReport must be a frozen dataclass."""
        report = decompose_cycle(balanced_df)
        with pytest.raises(FrozenInstanceError):
            report.bottleneck_stage = "return"  # type: ignore[misc]

    def test_input_dataframe_not_mutated(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """decompose_cycle must not mutate the input DataFrame."""
        snapshot = balanced_df.copy(deep=True)
        _ = decompose_cycle(balanced_df)
        pd.testing.assert_frame_equal(balanced_df, snapshot)


# ---------------------------------------------------------------------------
# Test 7 -- DataFrame round-trip
# ---------------------------------------------------------------------------


class TestDataFrameRoundTrip:
    """to_dataframe export helper."""

    def test_to_dataframe_columns(self, balanced_df: pd.DataFrame) -> None:
        """Exported DataFrame must have the documented column set."""
        report = decompose_cycle(balanced_df)
        exported = report.to_dataframe()
        expected = {
            "stage",
            "column",
            "n_cycles",
            "mean_min",
            "median_min",
            "p95_min",
            "min_min",
            "max_min",
            "std_min",
            "share_of_cycle",
        }
        assert set(exported.columns) == expected

    def test_to_dataframe_row_count_matches_stages(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """Exported row count must equal the number of stages in the report."""
        report = decompose_cycle(balanced_df)
        exported = report.to_dataframe()
        assert len(exported) == len(report.stage_stats)

    def test_to_dataframe_empty_report(self) -> None:
        """Empty report -> empty DataFrame with correct column headers."""
        empty_df = decompose_cycle(pd.DataFrame()).to_dataframe()
        assert empty_df.empty
        assert "stage" in empty_df.columns

    def test_to_dataframe_stage_order(
        self, balanced_df: pd.DataFrame
    ) -> None:
        """Exported DataFrame must be ordered by STAGE_ORDER."""
        report = decompose_cycle(balanced_df)
        exported = report.to_dataframe()
        assert exported["stage"].tolist() == list(STAGE_ORDER)


# ---------------------------------------------------------------------------
# Test 8 -- Sample data integration
# ---------------------------------------------------------------------------


class TestSampleDataIntegration:
    """End-to-end against the bundled demo/sample_data.csv."""

    def test_decompose_on_sample_data(self) -> None:
        """Decomposing the sample CSV must produce all five stages."""
        sample_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not sample_path.exists():
            pytest.skip("demo/sample_data.csv not found")
        df = pd.read_csv(sample_path)
        report = decompose_cycle(df)
        # Sample data uses timestamp-schema aliases.
        assert set(report.stage_stats.keys()) >= {
            "load",
            "haul",
            "dump",
            "return",
            "queue",
        }

    def test_sample_data_bottleneck_is_haul(self) -> None:
        """Realistic mining data has hauling as the bottleneck stage."""
        sample_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not sample_path.exists():
            pytest.skip("demo/sample_data.csv not found")
        df = pd.read_csv(sample_path)
        report = decompose_cycle(df)
        assert report.bottleneck_stage == "haul"

    def test_sample_data_total_cycles_matches_rowcount(self) -> None:
        """total_cycles should equal the CSV row count."""
        sample_path = Path(__file__).parent.parent / "demo" / "sample_data.csv"
        if not sample_path.exists():
            pytest.skip("demo/sample_data.csv not found")
        df = pd.read_csv(sample_path)
        report = decompose_cycle(df)
        assert report.total_cycles == len(df)
