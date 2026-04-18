"""Outlier filter for haul truck cycle records.

This module identifies and removes abnormal cycle records from a DataFrame so
that downstream KPIs (mean cycle time, productivity, match factor) are not
skewed by data-entry errors, stalled trucks, or one-off disruptions.

Two detection strategies are provided:

*IQR method* (``method="iqr"``)
    Robust against non-Gaussian data. A record is flagged when its value lies
    outside ``[Q1 - k * IQR, Q3 + k * IQR]`` where ``k`` defaults to ``1.5``.

*Z-score method* (``method="zscore"``)
    Assumes approximately normal distribution. A record is flagged when its
    absolute z-score exceeds a threshold (default ``3.0``).

Both methods operate per column, never mutate the input DataFrame, and return
a new DataFrame plus a diagnostic :class:`OutlierReport`. Rows with NaN values
in the target column are always flagged as invalid.

Design notes
------------
- Pure functions, no in-place mutation of input DataFrames.
- Frozen dataclasses for immutable result objects.
- Fail-fast validation with clear error messages on bad input.
- Mirrors the validation / result-object style of
  :mod:`queue_time_analyzer` and :mod:`fleet_match_factor_calculator`.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default IQR multiplier. 1.5 matches the classic Tukey fence.
DEFAULT_IQR_MULTIPLIER: float = 1.5

#: Default z-score cutoff. 3.0 corresponds to ~99.7 % of a normal distribution.
DEFAULT_ZSCORE_THRESHOLD: float = 3.0

_SUPPORTED_METHODS: Tuple[str, ...] = ("iqr", "zscore")


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlierReport:
    """Diagnostic summary produced by :func:`filter_outliers`.

    Attributes
    ----------
    column:
        Column the filter was applied to.
    method:
        Detection strategy used (``"iqr"`` or ``"zscore"``).
    lower_bound:
        Inclusive lower bound below which a value is flagged as an outlier.
    upper_bound:
        Inclusive upper bound above which a value is flagged as an outlier.
    n_total:
        Row count of the input DataFrame.
    n_outliers:
        Number of rows flagged as outliers (including NaN rows).
    n_kept:
        Number of rows retained after filtering (``n_total - n_outliers``).
    """

    column: str
    method: str
    lower_bound: float
    upper_bound: float
    n_total: int
    n_outliers: int
    n_kept: int

    @property
    def outlier_ratio(self) -> float:
        """Fraction of rows flagged as outliers. Returns ``0.0`` when empty."""
        if self.n_total == 0:
            return 0.0
        return self.n_outliers / self.n_total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    df: pd.DataFrame, column: str, method: str
) -> None:
    """Fail fast on obviously bad inputs.

    Parameters
    ----------
    df:
        Input DataFrame to validate.
    column:
        Column name that must exist and be numeric.
    method:
        Detection method that must be one of :data:`_SUPPORTED_METHODS`.

    Raises
    ------
    TypeError
        When ``df`` is not a :class:`pandas.DataFrame`.
    ValueError
        When ``column`` is missing, not numeric, or ``method`` is unsupported.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")
    if column not in df.columns:
        raise ValueError(
            f"column {column!r} not found in DataFrame (have: {list(df.columns)})"
        )
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"column {column!r} must be numeric, got dtype {df[column].dtype}"
        )
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {_SUPPORTED_METHODS}, got {method!r}"
        )


def _iqr_bounds(
    series: pd.Series, multiplier: float
) -> Tuple[float, float]:
    """Compute Tukey IQR fences for a numeric series.

    NaN values are excluded from the quantile computation. When the series
    is empty after dropping NaNs the bounds collapse to ``(-inf, +inf)`` so
    no rows are flagged.
    """
    clean = series.dropna()
    if clean.empty:
        return (float("-inf"), float("inf"))
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    return (q1 - multiplier * iqr, q3 + multiplier * iqr)


def _zscore_bounds(
    series: pd.Series, threshold: float
) -> Tuple[float, float]:
    """Compute symmetric z-score bounds for a numeric series.

    Falls back to ``(-inf, +inf)`` when the standard deviation is zero or
    the series is empty, so constant columns never trigger spurious flags.
    """
    clean = series.dropna()
    if clean.empty:
        return (float("-inf"), float("inf"))
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    if std == 0 or np.isnan(std):
        return (float("-inf"), float("inf"))
    return (mean - threshold * std, mean + threshold * std)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_outliers(
    df: pd.DataFrame,
    column: str = "total_cycle_min",
    method: str = "iqr",
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> Tuple[pd.DataFrame, OutlierReport]:
    """Return a new DataFrame with outlier rows removed.

    Parameters
    ----------
    df:
        Input cycle records. The caller retains ownership and the DataFrame
        is never mutated.
    column:
        Numeric column on which to detect outliers (defaults to
        ``"total_cycle_min"``).
    method:
        ``"iqr"`` (default, robust) or ``"zscore"`` (assumes normal-ish).
    iqr_multiplier:
        Tukey fence multiplier when ``method="iqr"``. Must be non-negative.
    zscore_threshold:
        Cutoff when ``method="zscore"``. Must be positive.

    Returns
    -------
    tuple
        ``(filtered_df, report)`` where ``filtered_df`` is a new DataFrame
        containing only non-outlier rows, and ``report`` is an immutable
        :class:`OutlierReport` summarising the operation.

    Raises
    ------
    TypeError
        When ``df`` is not a DataFrame.
    ValueError
        When the column is missing / non-numeric, the method is unsupported,
        or a threshold is not strictly positive.

    Notes
    -----
    - Rows containing NaN in ``column`` are always treated as outliers and
      dropped, because they cannot contribute to a meaningful statistic.
    - The returned DataFrame preserves the original index and column order.
    - For a constant column (zero variance) no rows are flagged, which is
      the desired behaviour for degenerate datasets.
    """
    _validate_inputs(df, column, method)
    if iqr_multiplier < 0:
        raise ValueError(
            f"iqr_multiplier must be non-negative, got {iqr_multiplier}"
        )
    if zscore_threshold <= 0:
        raise ValueError(
            f"zscore_threshold must be positive, got {zscore_threshold}"
        )

    n_total = len(df)
    if n_total == 0:
        report = OutlierReport(
            column=column,
            method=method,
            lower_bound=float("-inf"),
            upper_bound=float("inf"),
            n_total=0,
            n_outliers=0,
            n_kept=0,
        )
        return df.copy(), report

    series = df[column]
    if method == "iqr":
        lower, upper = _iqr_bounds(series, iqr_multiplier)
    else:  # method == "zscore"; guarded by _validate_inputs
        lower, upper = _zscore_bounds(series, zscore_threshold)

    # NaN rows are always excluded; inclusive bounds.
    in_range = series.between(lower, upper, inclusive="both") & series.notna()
    filtered = df.loc[in_range].copy()

    n_kept = int(in_range.sum())
    n_outliers = n_total - n_kept

    report = OutlierReport(
        column=column,
        method=method,
        lower_bound=float(lower),
        upper_bound=float(upper),
        n_total=n_total,
        n_outliers=n_outliers,
        n_kept=n_kept,
    )
    return filtered, report


def flag_outliers(
    df: pd.DataFrame,
    column: str = "total_cycle_min",
    method: str = "iqr",
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    flag_column: str = "is_outlier",
) -> pd.DataFrame:
    """Return a new DataFrame with an added boolean outlier flag column.

    Unlike :func:`filter_outliers`, this function keeps every row so callers
    can review or visualise the flagged records before deciding to drop them.

    Parameters
    ----------
    df:
        Input cycle records (not mutated).
    column:
        Numeric column to evaluate.
    method:
        ``"iqr"`` or ``"zscore"``.
    iqr_multiplier, zscore_threshold:
        Same semantics as :func:`filter_outliers`.
    flag_column:
        Name of the new boolean column. Overwritten if it already exists.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with an extra boolean column ``flag_column`` set to
        ``True`` for outliers (including NaN rows) and ``False`` otherwise.
    """
    _validate_inputs(df, column, method)
    if iqr_multiplier < 0:
        raise ValueError(
            f"iqr_multiplier must be non-negative, got {iqr_multiplier}"
        )
    if zscore_threshold <= 0:
        raise ValueError(
            f"zscore_threshold must be positive, got {zscore_threshold}"
        )

    out = df.copy()
    if len(out) == 0:
        out[flag_column] = pd.Series([], dtype=bool)
        return out

    series = out[column]
    if method == "iqr":
        lower, upper = _iqr_bounds(series, iqr_multiplier)
    else:
        lower, upper = _zscore_bounds(series, zscore_threshold)

    in_range = series.between(lower, upper, inclusive="both") & series.notna()
    out[flag_column] = ~in_range
    return out


__all__ = [
    "DEFAULT_IQR_MULTIPLIER",
    "DEFAULT_ZSCORE_THRESHOLD",
    "OutlierReport",
    "filter_outliers",
    "flag_outliers",
]
