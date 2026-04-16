"""
Fleet Match Factor Calculator for open-pit mining haul truck operations.

The **match factor** (MF) quantifies the balance between loader capacity and
truck fleet capacity.  A value of 1.0 means the loader and trucks are perfectly
paired; MF < 1 indicates the loader is the bottleneck (trucks are waiting); MF
> 1 indicates the trucks are the bottleneck (the loader sits idle between
servicing cycles).

Formula
-------
::

    MF = (n_trucks * loader_cycle_time_min) / (truck_cycle_time_min * n_passes)

where ``n_passes`` is the number of loader passes required to fill one truck
(typically 1 for large electric rope shovels, 2–5 for hydraulic excavators).

Reference
---------
Atkinson, T. (1992). *Selection and sizing of excavating equipment*.
Surface Mining, 2nd ed., SME, pp. 503–514.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Match factor threshold below which a fleet is considered under-trucked.
UNDER_TRUCKED_THRESHOLD: float = 0.90

#: Match factor threshold above which a fleet is considered over-trucked.
OVER_TRUCKED_THRESHOLD: float = 1.10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PitMatchResult:
    """Immutable result of a match factor calculation for one pit.

    Attributes
    ----------
    pit_name:
        Identifier of the pit or loading zone (e.g. ``"North Pit"``).
    n_trucks:
        Number of trucks assigned to this pit during the observed period.
    avg_truck_cycle_min:
        Mean full truck cycle time (load + haul + dump + return) in minutes.
    loader_cycle_time_min:
        Effective loader cycle time per truck fill in minutes.
    n_passes:
        Number of loader passes required to fill one truck.
    match_factor:
        Computed match factor (dimensionless).  Values are rounded to four
        decimal places.
    condition:
        Human-readable label: ``"balanced"``, ``"under-trucked"``, or
        ``"over-trucked"``.
    """

    pit_name: str
    n_trucks: int
    avg_truck_cycle_min: float
    loader_cycle_time_min: float
    n_passes: int
    match_factor: float
    condition: str


@dataclass(frozen=True)
class MatchFactorReport:
    """Immutable fleet match factor report across all pits.

    Attributes
    ----------
    pit_results:
        Ordered sequence of :class:`PitMatchResult` objects, one per pit.
    overall_match_factor:
        Fleet-wide weighted mean match factor (weighted by truck count per
        pit).  Returns ``float("nan")`` when no valid pit results exist.
    n_under_trucked:
        Number of pits classified as under-trucked.
    n_over_trucked:
        Number of pits classified as over-trucked.
    n_balanced:
        Number of pits classified as balanced.
    """

    pit_results: tuple[PitMatchResult, ...]
    overall_match_factor: float
    n_under_trucked: int
    n_over_trucked: int
    n_balanced: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return pit results as a tidy :class:`pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per pit with columns mirroring :class:`PitMatchResult`
            attributes.  Returns an empty DataFrame when no pit results exist.

        Examples
        --------
        >>> report.to_dataframe().columns.tolist()
        ['pit_name', 'n_trucks', 'avg_truck_cycle_min', 'loader_cycle_time_min',
         'n_passes', 'match_factor', 'condition']
        """
        if not self.pit_results:
            return pd.DataFrame()

        rows = [
            {
                "pit_name": r.pit_name,
                "n_trucks": r.n_trucks,
                "avg_truck_cycle_min": r.avg_truck_cycle_min,
                "loader_cycle_time_min": r.loader_cycle_time_min,
                "n_passes": r.n_passes,
                "match_factor": r.match_factor,
                "condition": r.condition,
            }
            for r in self.pit_results
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure calculation helpers
# ---------------------------------------------------------------------------


def _classify_condition(match_factor: float) -> str:
    """Return a human-readable fleet balance label for *match_factor*.

    Parameters
    ----------
    match_factor:
        Computed match factor value.

    Returns
    -------
    str
        ``"under-trucked"`` when MF < ``UNDER_TRUCKED_THRESHOLD``,
        ``"over-trucked"`` when MF > ``OVER_TRUCKED_THRESHOLD``, otherwise
        ``"balanced"``.

    Examples
    --------
    >>> _classify_condition(0.85)
    'under-trucked'
    >>> _classify_condition(1.05)
    'balanced'
    >>> _classify_condition(1.20)
    'over-trucked'
    """
    if match_factor < UNDER_TRUCKED_THRESHOLD:
        return "under-trucked"
    if match_factor > OVER_TRUCKED_THRESHOLD:
        return "over-trucked"
    return "balanced"


def compute_match_factor(
    n_trucks: int,
    truck_cycle_time_min: float,
    loader_cycle_time_min: float,
    n_passes: int = 1,
) -> float:
    """Compute the loader-to-truck match factor for a single pit.

    The match factor is the ratio of loader throughput capacity to truck fleet
    throughput capacity over the same time window:

    ::

        MF = (n_trucks * loader_cycle_time_min) / (truck_cycle_time_min * n_passes)

    Parameters
    ----------
    n_trucks:
        Number of trucks assigned to serve the loader.  Must be >= 1.
    truck_cycle_time_min:
        Mean full truck cycle time in minutes (loading + haul + dump +
        return).  Must be > 0.
    loader_cycle_time_min:
        Time for the loader to complete one pass in minutes.  Must be > 0.
    n_passes:
        Number of loader passes needed to fill one truck.  Defaults to ``1``
        for large electric rope shovels.  Must be >= 1.

    Returns
    -------
    float
        Match factor rounded to four decimal places.

    Raises
    ------
    ValueError
        When any of the following conditions hold:

        - ``n_trucks < 1``
        - ``truck_cycle_time_min <= 0``
        - ``loader_cycle_time_min <= 0``
        - ``n_passes < 1``

    Examples
    --------
    >>> compute_match_factor(n_trucks=4, truck_cycle_time_min=50.0,
    ...                      loader_cycle_time_min=12.0, n_passes=1)
    0.96
    >>> compute_match_factor(n_trucks=5, truck_cycle_time_min=50.0,
    ...                      loader_cycle_time_min=12.0, n_passes=1)
    1.2
    """
    if n_trucks < 1:
        raise ValueError(f"n_trucks must be >= 1, got {n_trucks}.")
    if truck_cycle_time_min <= 0:
        raise ValueError(
            f"truck_cycle_time_min must be > 0, got {truck_cycle_time_min}."
        )
    if loader_cycle_time_min <= 0:
        raise ValueError(
            f"loader_cycle_time_min must be > 0, got {loader_cycle_time_min}."
        )
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}.")

    mf = (n_trucks * loader_cycle_time_min) / (truck_cycle_time_min * n_passes)
    return round(mf, 4)


# ---------------------------------------------------------------------------
# DataFrame-level calculator
# ---------------------------------------------------------------------------


def calculate_fleet_match_factor(
    df: pd.DataFrame,
    loader_cycle_time_min: float,
    n_passes: int = 1,
    pit_col: str = "pit_name",
    truck_col: str = "truck_id",
    cycle_time_col: Optional[str] = None,
) -> MatchFactorReport:
    """Compute match factor per pit from a cycle DataFrame.

    For each unique pit in *df*, the function counts distinct trucks and
    computes the mean truck cycle time, then calls :func:`compute_match_factor`
    to produce a :class:`PitMatchResult`.  Results are assembled into an
    immutable :class:`MatchFactorReport`.

    Parameters
    ----------
    df:
        Preprocessed/enriched cycle DataFrame.  Must contain at minimum
        columns for ``pit_col`` and ``truck_col``.
    loader_cycle_time_min:
        Effective loader swing-and-load cycle time per pass in minutes.
        Must be > 0.
    n_passes:
        Number of loader passes per truck fill.  Defaults to ``1``.
    pit_col:
        Name of the column identifying the loading pit or zone.
        Defaults to ``"pit_name"``.
    truck_col:
        Name of the column identifying truck IDs.
        Defaults to ``"truck_id"``.
    cycle_time_col:
        Name of the column holding per-cycle total time in minutes.  When
        ``None`` (default) the function tries ``"total_cycle_min"`` then
        ``"computed_cycle_min"`` in that order.

    Returns
    -------
    MatchFactorReport
        Immutable report with per-pit results and fleet-wide summary.
        Returns a report with empty ``pit_results`` when *df* is empty or
        required columns are absent.

    Raises
    ------
    ValueError
        When ``loader_cycle_time_min <= 0`` or ``n_passes < 1``.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.fleet_match_factor_calculator import calculate_fleet_match_factor
    >>> df = pd.DataFrame({
    ...     "truck_id":       ["T1", "T1", "T2", "T2", "T3"],
    ...     "pit_name":       ["North"] * 5,
    ...     "total_cycle_min": [50.0, 52.0, 48.0, 51.0, 49.0],
    ... })
    >>> report = calculate_fleet_match_factor(df, loader_cycle_time_min=12.0)
    >>> report.pit_results[0].n_trucks
    3
    >>> report.pit_results[0].condition
    'under-trucked'
    """
    if loader_cycle_time_min <= 0:
        raise ValueError(
            f"loader_cycle_time_min must be > 0, got {loader_cycle_time_min}."
        )
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}.")

    # Return empty report for empty or incomplete input
    if df.empty or pit_col not in df.columns or truck_col not in df.columns:
        return MatchFactorReport(
            pit_results=(),
            overall_match_factor=float("nan"),
            n_under_trucked=0,
            n_over_trucked=0,
            n_balanced=0,
        )

    # Resolve the cycle time column
    if cycle_time_col is not None:
        effective_cycle_col: Optional[str] = cycle_time_col
    elif "total_cycle_min" in df.columns:
        effective_cycle_col = "total_cycle_min"
    elif "computed_cycle_min" in df.columns:
        effective_cycle_col = "computed_cycle_min"
    else:
        effective_cycle_col = None

    pit_results: List[PitMatchResult] = []

    for pit_name, pit_df in df.groupby(pit_col, sort=True):
        n_trucks = int(pit_df[truck_col].nunique())
        if n_trucks < 1:
            continue

        if effective_cycle_col and effective_cycle_col in pit_df.columns:
            valid_times = pd.to_numeric(
                pit_df[effective_cycle_col], errors="coerce"
            ).dropna()
            valid_times = valid_times[valid_times > 0]
            avg_cycle = float(valid_times.mean()) if not valid_times.empty else 0.0
        else:
            avg_cycle = 0.0

        if avg_cycle <= 0:
            # Cannot compute a meaningful MF without valid cycle times
            continue

        mf = compute_match_factor(
            n_trucks=n_trucks,
            truck_cycle_time_min=avg_cycle,
            loader_cycle_time_min=loader_cycle_time_min,
            n_passes=n_passes,
        )
        condition = _classify_condition(mf)

        pit_results.append(
            PitMatchResult(
                pit_name=str(pit_name),
                n_trucks=n_trucks,
                avg_truck_cycle_min=round(avg_cycle, 4),
                loader_cycle_time_min=loader_cycle_time_min,
                n_passes=n_passes,
                match_factor=mf,
                condition=condition,
            )
        )

    # Fleet-wide weighted mean match factor
    if pit_results:
        total_trucks = sum(r.n_trucks for r in pit_results)
        weighted_sum = sum(r.match_factor * r.n_trucks for r in pit_results)
        overall_mf = round(weighted_sum / total_trucks, 4) if total_trucks > 0 else float("nan")
    else:
        overall_mf = float("nan")

    n_under = sum(1 for r in pit_results if r.condition == "under-trucked")
    n_over = sum(1 for r in pit_results if r.condition == "over-trucked")
    n_bal = sum(1 for r in pit_results if r.condition == "balanced")

    return MatchFactorReport(
        pit_results=tuple(pit_results),
        overall_match_factor=overall_mf,
        n_under_trucked=n_under,
        n_over_trucked=n_over,
        n_balanced=n_bal,
    )
