"""Queue Time Analyzer for open-pit mining haul truck operations.

This module quantifies how much of each truck cycle is consumed by queueing at
the loading face (or any other servicing point) and identifies the worst
offenders.  Excessive queue time is one of the strongest signals that a fleet
is over-trucked relative to loader capacity, that shovel availability is poor,
or that dispatch routing is unbalanced.

Definitions
-----------
*Queue ratio*
    ``queue_time_min / total_cycle_min`` -- the fraction of the cycle a truck
    spends waiting in queue, expressed as a number in ``[0, 1]``.

*Severity bucket*
    ``"low"`` (<= 5 % of cycle), ``"moderate"`` (5-15 %), ``"high"``
    (15-25 %), ``"critical"`` (> 25 %).  Thresholds follow industry rules of
    thumb for open-pit operations where loader queueing above ~15 % of cycle
    typically warrants dispatch intervention.

Design notes
------------
- Pure functions, no in-place mutation of input DataFrames.
- Frozen dataclasses for immutable result objects.
- Fail-fast validation with clear error messages on bad input.
- Skips rows with missing/non-positive cycle time rather than rejecting the
  whole DataFrame, mirroring :mod:`fleet_match_factor_calculator`.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Upper bound (inclusive) of the "low" severity bucket as a queue ratio.
LOW_SEVERITY_MAX: float = 0.05

#: Upper bound (inclusive) of the "moderate" severity bucket as a queue ratio.
MODERATE_SEVERITY_MAX: float = 0.15

#: Upper bound (inclusive) of the "high" severity bucket as a queue ratio.
HIGH_SEVERITY_MAX: float = 0.25

_SEVERITY_LABELS: Tuple[str, ...] = ("low", "moderate", "high", "critical")


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TruckQueueStats:
    """Per-truck queue-time roll-up.

    Attributes:
        truck_id: Truck identifier (e.g. ``"TRK01"``).
        n_cycles: Number of valid cycles included in the aggregate.
        total_queue_min: Sum of queue minutes across all cycles.
        total_cycle_min: Sum of cycle minutes across all cycles.
        avg_queue_min: Mean queue minutes per cycle.
        queue_ratio: ``total_queue_min / total_cycle_min``, in ``[0, 1]``.
        severity: Severity bucket for this truck's queue ratio.
    """

    truck_id: str
    n_cycles: int
    total_queue_min: float
    total_cycle_min: float
    avg_queue_min: float
    queue_ratio: float
    severity: str


@dataclass(frozen=True)
class QueueTimeReport:
    """Fleet-wide queue-time report.

    Attributes:
        truck_stats: Tuple of :class:`TruckQueueStats`, sorted by descending
            ``queue_ratio`` so the worst offenders appear first.
        total_queue_min: Sum of queue minutes across all valid cycles.
        total_cycle_min: Sum of cycle minutes across all valid cycles.
        fleet_queue_ratio: ``total_queue_min / total_cycle_min`` for the whole
            fleet (``float("nan")`` when no valid cycles exist).
        worst_truck: Truck ID with the highest ``queue_ratio`` (``None`` when
            the report is empty).
        n_critical_trucks: Number of trucks in the ``"critical"`` bucket.
    """

    truck_stats: Tuple[TruckQueueStats, ...]
    total_queue_min: float
    total_cycle_min: float
    fleet_queue_ratio: float
    worst_truck: Optional[str]
    n_critical_trucks: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-truck stats as a tidy :class:`pandas.DataFrame`.

        Returns:
            One row per truck, ordered by descending ``queue_ratio``.  Empty
            DataFrame when no truck stats are present.
        """
        if not self.truck_stats:
            return pd.DataFrame(
                columns=[
                    "truck_id",
                    "n_cycles",
                    "total_queue_min",
                    "total_cycle_min",
                    "avg_queue_min",
                    "queue_ratio",
                    "severity",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "truck_id": s.truck_id,
                    "n_cycles": s.n_cycles,
                    "total_queue_min": s.total_queue_min,
                    "total_cycle_min": s.total_cycle_min,
                    "avg_queue_min": s.avg_queue_min,
                    "queue_ratio": s.queue_ratio,
                    "severity": s.severity,
                }
                for s in self.truck_stats
            ]
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def classify_queue_severity(queue_ratio: float) -> str:
    """Classify a queue ratio into a severity bucket.

    Args:
        queue_ratio: Fraction of cycle spent queueing, in ``[0, 1]``.

    Returns:
        ``"low"``, ``"moderate"``, ``"high"``, or ``"critical"``.

    Raises:
        ValueError: If *queue_ratio* is negative, greater than 1, or NaN.

    Examples:
        >>> classify_queue_severity(0.03)
        'low'
        >>> classify_queue_severity(0.20)
        'high'
        >>> classify_queue_severity(0.40)
        'critical'
    """
    if queue_ratio != queue_ratio:  # NaN check
        raise ValueError("queue_ratio must not be NaN.")
    if queue_ratio < 0 or queue_ratio > 1:
        raise ValueError(
            f"queue_ratio must be in [0, 1], got {queue_ratio}."
        )
    if queue_ratio <= LOW_SEVERITY_MAX:
        return _SEVERITY_LABELS[0]
    if queue_ratio <= MODERATE_SEVERITY_MAX:
        return _SEVERITY_LABELS[1]
    if queue_ratio <= HIGH_SEVERITY_MAX:
        return _SEVERITY_LABELS[2]
    return _SEVERITY_LABELS[3]


def _resolve_cycle_time_col(
    df: pd.DataFrame, cycle_time_col: Optional[str]
) -> Optional[str]:
    """Return the first available cycle-time column or *None*."""
    if cycle_time_col is not None:
        return cycle_time_col if cycle_time_col in df.columns else None
    for candidate in ("total_cycle_min", "computed_cycle_min"):
        if candidate in df.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------


def analyze_queue_time(
    df: pd.DataFrame,
    truck_col: str = "truck_id",
    queue_col: str = "queue_time_min",
    cycle_time_col: Optional[str] = None,
) -> QueueTimeReport:
    """Compute per-truck and fleet-wide queue-time statistics.

    For each truck the function aggregates queue minutes and cycle minutes
    across valid cycles, then derives the queue ratio and severity bucket.
    Rows with non-positive cycle time, negative queue time, or missing values
    in the required columns are silently skipped (consistent with the rest of
    this package).

    Args:
        df: Cycle DataFrame.  Must contain *truck_col* and *queue_col* and at
            least one of ``"total_cycle_min"`` or ``"computed_cycle_min"``
            (or an explicit *cycle_time_col*).
        truck_col: Name of the truck identifier column.  Defaults to
            ``"truck_id"``.
        queue_col: Name of the per-cycle queue-time column (minutes).
            Defaults to ``"queue_time_min"``.
        cycle_time_col: Name of the per-cycle total-time column (minutes).
            When ``None`` (default) ``"total_cycle_min"`` is used if present,
            otherwise ``"computed_cycle_min"``.

    Returns:
        Immutable :class:`QueueTimeReport`.  When *df* is empty or required
        columns are absent, the report has empty ``truck_stats`` and
        ``fleet_queue_ratio == float('nan')``.

    Raises:
        TypeError: If *df* is not a :class:`pandas.DataFrame`.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "truck_id": ["T1", "T1", "T2"],
        ...     "queue_time_min": [3.0, 4.0, 12.0],
        ...     "total_cycle_min": [50.0, 60.0, 50.0],
        ... })
        >>> report = analyze_queue_time(df)
        >>> report.worst_truck
        'T2'
        >>> report.truck_stats[0].severity
        'high'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame, got {type(df).__name__}."
        )

    cycle_col = _resolve_cycle_time_col(df, cycle_time_col)
    required_cols = {truck_col, queue_col}
    if (
        df.empty
        or not required_cols.issubset(df.columns)
        or cycle_col is None
    ):
        return QueueTimeReport(
            truck_stats=(),
            total_queue_min=0.0,
            total_cycle_min=0.0,
            fleet_queue_ratio=float("nan"),
            worst_truck=None,
            n_critical_trucks=0,
        )

    # Build a sanitized working copy -- no mutation of the input DataFrame.
    work = df[[truck_col, queue_col, cycle_col]].copy()
    work[queue_col] = pd.to_numeric(work[queue_col], errors="coerce")
    work[cycle_col] = pd.to_numeric(work[cycle_col], errors="coerce")
    work = work.dropna(subset=[truck_col, queue_col, cycle_col])
    work = work[(work[cycle_col] > 0) & (work[queue_col] >= 0)]
    # Clamp pathological queue > cycle to cycle (queue can never exceed cycle).
    over_cap = work[queue_col] > work[cycle_col]
    if over_cap.any():
        work.loc[over_cap, queue_col] = work.loc[over_cap, cycle_col]

    if work.empty:
        return QueueTimeReport(
            truck_stats=(),
            total_queue_min=0.0,
            total_cycle_min=0.0,
            fleet_queue_ratio=float("nan"),
            worst_truck=None,
            n_critical_trucks=0,
        )

    grouped = work.groupby(truck_col, sort=True)
    stats: list[TruckQueueStats] = []
    for truck_id, sub in grouped:
        n_cycles = int(len(sub))
        total_q = float(sub[queue_col].sum())
        total_c = float(sub[cycle_col].sum())
        avg_q = round(total_q / n_cycles, 4) if n_cycles else 0.0
        ratio = round(total_q / total_c, 4) if total_c > 0 else 0.0
        stats.append(
            TruckQueueStats(
                truck_id=str(truck_id),
                n_cycles=n_cycles,
                total_queue_min=round(total_q, 4),
                total_cycle_min=round(total_c, 4),
                avg_queue_min=avg_q,
                queue_ratio=ratio,
                severity=classify_queue_severity(ratio),
            )
        )

    stats.sort(key=lambda s: s.queue_ratio, reverse=True)

    fleet_q = sum(s.total_queue_min for s in stats)
    fleet_c = sum(s.total_cycle_min for s in stats)
    fleet_ratio = round(fleet_q / fleet_c, 4) if fleet_c > 0 else float("nan")
    worst = stats[0].truck_id if stats else None
    n_crit = sum(1 for s in stats if s.severity == "critical")

    return QueueTimeReport(
        truck_stats=tuple(stats),
        total_queue_min=round(fleet_q, 4),
        total_cycle_min=round(fleet_c, 4),
        fleet_queue_ratio=fleet_ratio,
        worst_truck=worst,
        n_critical_trucks=n_crit,
    )
