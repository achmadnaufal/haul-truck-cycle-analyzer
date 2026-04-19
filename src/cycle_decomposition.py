"""Cycle time decomposition and fleet-level bottleneck identification.

This module breaks each haul truck cycle into its constituent stages
(``load``, ``haul``, ``dump``, ``return``, ``queue``) and computes descriptive
statistics for each one.  Given those statistics, it then identifies the
fleet-wide bottleneck -- the stage that consumes the largest share of the
average cycle and is therefore the highest-leverage optimisation target.

The module is intentionally complementary to the per-row
:func:`src.main.identify_bottleneck` helper: that function answers "which stage
dominated *this* cycle?", while this module answers "which stage dominates the
*fleet*?".  In real operations the two can disagree: a single truck may spend
most of one cycle queueing while the fleet-wide bottleneck is hauling.

Design notes
------------
- Pure functions, no in-place mutation of input DataFrames.
- Frozen dataclasses for immutable result objects.
- Fail-fast validation with clear error messages on bad input.
- Graceful handling of empty DataFrames, missing stage columns, NaN values,
  and degenerate inputs (all zeros).
- Mirrors the validation and result-object style of
  :mod:`queue_time_analyzer`, :mod:`fleet_match_factor_calculator`, and
  :mod:`outlier_filter`.

Reference
---------
Caterpillar Global Mining, *Caterpillar Performance Handbook*, Edition 49,
Chapter "Haul Road Efficiency", for stage-time benchmarks.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical stage names in the order a haul truck traverses them.
STAGE_ORDER: Tuple[str, ...] = ("load", "haul", "dump", "return", "queue")

#: Mapping from canonical stage name to the set of column aliases accepted on
#: input.  The first entry in each tuple is the canonical column produced by
#: :meth:`src.main.HaulTruckAnalyzer.preprocess`.
_STAGE_COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "load": ("loading_time_min", "load_time_min"),
    "haul": ("hauling_time_min", "haul_time_min"),
    "dump": ("dumping_time_min", "dump_time_min"),
    "return": ("return_time_min",),
    "queue": ("queue_time_min",),
}

#: Percentile used for the ``p95`` summary statistic.
_P95_QUANTILE: float = 0.95


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageStats:
    """Immutable per-stage descriptive statistics.

    Attributes:
        stage: Canonical stage name (``"load"``, ``"haul"``, ``"dump"``,
            ``"return"``, or ``"queue"``).
        column: Source column name used to populate the statistic.
        n_cycles: Number of cycles contributing a finite, non-negative value
            to the stage.
        mean_min: Arithmetic mean of stage minutes across ``n_cycles``.
        median_min: Median (50th percentile) of stage minutes.
        p95_min: 95th percentile of stage minutes, useful for identifying
            long-tail outliers.
        min_min: Minimum stage minutes observed.
        max_min: Maximum stage minutes observed.
        std_min: Sample standard deviation of stage minutes (``ddof=1``).
            ``0.0`` when ``n_cycles < 2`` or values are constant.
        share_of_cycle: Fraction of the average total cycle that this stage
            consumes, in ``[0, 1]``.  Computed from the sum of stage means
            divided into this stage's mean.
    """

    stage: str
    column: str
    n_cycles: int
    mean_min: float
    median_min: float
    p95_min: float
    min_min: float
    max_min: float
    std_min: float
    share_of_cycle: float


@dataclass(frozen=True)
class CycleDecompositionReport:
    """Immutable fleet-level cycle decomposition result.

    Attributes:
        stage_stats: Mapping from canonical stage name to its
            :class:`StageStats`.  Only stages present in the input appear.
        total_cycles: Row count of the input DataFrame after dropping
            fully empty rows.
        mean_total_cycle_min: Sum of stage means -- equivalent to the mean
            cycle time when every cycle populates every stage.
        bottleneck_stage: Stage with the largest ``mean_min`` (and therefore
            the largest ``share_of_cycle``).  ``None`` when no stage data is
            available.
        bottleneck_share: ``share_of_cycle`` of the bottleneck stage, or
            ``0.0`` when no stage data is available.
    """

    stage_stats: Mapping[str, StageStats]
    total_cycles: int
    mean_total_cycle_min: float
    bottleneck_stage: Optional[str]
    bottleneck_share: float

    def to_dataframe(self) -> pd.DataFrame:
        """Return stage stats as a tidy :class:`pandas.DataFrame`.

        Returns:
            One row per stage ordered by :data:`STAGE_ORDER`.  Columns mirror
            the :class:`StageStats` attributes.  An empty DataFrame when the
            report contains no stage data.

        Examples:
            >>> report.to_dataframe().columns.tolist()
            ['stage', 'column', 'n_cycles', 'mean_min', 'median_min',
             'p95_min', 'min_min', 'max_min', 'std_min', 'share_of_cycle']
        """
        if not self.stage_stats:
            return pd.DataFrame(
                columns=[
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
                ]
            )
        rows = []
        for stage in STAGE_ORDER:
            s = self.stage_stats.get(stage)
            if s is None:
                continue
            rows.append(
                {
                    "stage": s.stage,
                    "column": s.column,
                    "n_cycles": s.n_cycles,
                    "mean_min": s.mean_min,
                    "median_min": s.median_min,
                    "p95_min": s.p95_min,
                    "min_min": s.min_min,
                    "max_min": s.max_min,
                    "std_min": s.std_min,
                    "share_of_cycle": s.share_of_cycle,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_stage_column(
    df_columns: "pd.Index | list[str]", stage: str
) -> Optional[str]:
    """Return the first alias present in *df_columns* for *stage*, else None.

    Args:
        df_columns: Index or list of column names to search.
        stage: Canonical stage name from :data:`STAGE_ORDER`.

    Returns:
        Column name to use for the stage, or ``None`` when no alias is
        present.
    """
    aliases = _STAGE_COLUMN_ALIASES.get(stage, ())
    for alias in aliases:
        if alias in df_columns:
            return alias
    return None


def _sanitize_series(series: pd.Series) -> pd.Series:
    """Coerce to numeric, drop NaN, and drop negatives.

    Negative durations are impossible, so they are treated as missing rather
    than clamped to zero.  This keeps the resulting statistics honest (a
    data-entry error should not silently lower the mean).
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return numeric[numeric >= 0]


def _round4(value: float) -> float:
    """Round a float to four decimal places, preserving NaN as 0.0."""
    if value != value:  # NaN check
        return 0.0
    return round(float(value), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_cycle(df: pd.DataFrame) -> CycleDecompositionReport:
    """Compute per-stage statistics and the fleet-level bottleneck stage.

    For each canonical stage (``load``, ``haul``, ``dump``, ``return``,
    ``queue``) the function builds a :class:`StageStats` record containing
    mean, median, p95, min, max, standard deviation, and the share of the
    average cycle consumed by the stage.  The stage with the highest mean is
    reported as the fleet bottleneck.

    Both standard (``loading_time_min``) and timestamp-schema
    (``load_time_min``) column names are accepted.  NaN values and negative
    durations are dropped per stage.  Stages missing from the input are
    silently skipped rather than raising.

    Args:
        df: Cycle DataFrame.  Must contain at least one of the canonical
            stage columns listed in :data:`_STAGE_COLUMN_ALIASES`.

    Returns:
        Immutable :class:`CycleDecompositionReport`.  When *df* is empty or
        no stage columns are present, the report has empty ``stage_stats``,
        ``mean_total_cycle_min == 0.0``, and ``bottleneck_stage is None``.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "truck_id":        ["T1", "T2"],
        ...     "loading_time_min": [8.0, 9.0],
        ...     "hauling_time_min": [20.0, 22.0],
        ...     "dumping_time_min": [3.0, 3.5],
        ...     "return_time_min":  [17.0, 18.0],
        ...     "queue_time_min":   [4.0, 5.0],
        ... })
        >>> report = decompose_cycle(df)
        >>> report.bottleneck_stage
        'haul'
        >>> report.stage_stats["haul"].mean_min
        21.0
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )

    # Drop fully empty rows so they do not pollute n_cycles counts.
    clean = df.dropna(how="all")

    if clean.empty:
        return CycleDecompositionReport(
            stage_stats={},
            total_cycles=0,
            mean_total_cycle_min=0.0,
            bottleneck_stage=None,
            bottleneck_share=0.0,
        )

    # First pass: compute mean per stage so we can derive share_of_cycle.
    stage_to_column: Dict[str, str] = {}
    stage_means: Dict[str, float] = {}

    for stage in STAGE_ORDER:
        column = _resolve_stage_column(clean.columns, stage)
        if column is None:
            continue
        values = _sanitize_series(clean[column])
        if values.empty:
            continue
        stage_to_column[stage] = column
        stage_means[stage] = float(values.mean())

    if not stage_means:
        return CycleDecompositionReport(
            stage_stats={},
            total_cycles=int(len(clean)),
            mean_total_cycle_min=0.0,
            bottleneck_stage=None,
            bottleneck_share=0.0,
        )

    total_mean_cycle = sum(stage_means.values())

    # Second pass: full stats using the pre-computed means for share_of_cycle.
    stage_stats: Dict[str, StageStats] = {}
    for stage, column in stage_to_column.items():
        values = _sanitize_series(clean[column])
        n = int(values.shape[0])
        mean_v = stage_means[stage]
        share = (mean_v / total_mean_cycle) if total_mean_cycle > 0 else 0.0
        std_v = float(values.std(ddof=1)) if n > 1 else 0.0
        stage_stats[stage] = StageStats(
            stage=stage,
            column=column,
            n_cycles=n,
            mean_min=_round4(mean_v),
            median_min=_round4(values.median()),
            p95_min=_round4(values.quantile(_P95_QUANTILE)),
            min_min=_round4(values.min()),
            max_min=_round4(values.max()),
            std_min=_round4(std_v),
            share_of_cycle=_round4(share),
        )

    bottleneck_stage = max(stage_means, key=lambda s: stage_means[s])
    bottleneck_share = stage_stats[bottleneck_stage].share_of_cycle

    return CycleDecompositionReport(
        stage_stats=stage_stats,
        total_cycles=int(len(clean)),
        mean_total_cycle_min=_round4(total_mean_cycle),
        bottleneck_stage=bottleneck_stage,
        bottleneck_share=bottleneck_share,
    )


def rank_stages_by_median(df: pd.DataFrame) -> Tuple[Tuple[str, float], ...]:
    """Return stages ranked by descending median stage time.

    Useful when the fleet has heavy-tailed stage distributions (e.g. a few
    very long queues) and the caller wants the typical-case bottleneck rather
    than the mean-based bottleneck.

    Args:
        df: Cycle DataFrame.  Accepts both the standard and timestamp
            schemas; see :func:`decompose_cycle` for accepted column names.

    Returns:
        Tuple of ``(stage, median_min)`` pairs ordered from largest to
        smallest median.  Empty tuple when no stage data is available.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "loading_time_min": [8.0, 9.0],
        ...     "hauling_time_min": [20.0, 22.0],
        ...     "return_time_min":  [17.0, 18.0],
        ... })
        >>> rank_stages_by_median(df)[0][0]
        'haul'
    """
    report = decompose_cycle(df)
    if not report.stage_stats:
        return ()
    pairs = [
        (stats.stage, stats.median_min)
        for stats in report.stage_stats.values()
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return tuple(pairs)


__all__ = [
    "STAGE_ORDER",
    "CycleDecompositionReport",
    "StageStats",
    "decompose_cycle",
    "rank_stages_by_median",
]
