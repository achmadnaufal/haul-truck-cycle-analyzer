"""
Haul truck cycle time analysis, bottleneck identification, and optimization.

This module provides the core HaulTruckAnalyzer class for processing mining
haul truck cycle data, computing productivity KPIs, identifying bottlenecks,
and generating fleet-level summaries.

Supported input schemas
-----------------------
The analyzer accepts two common column layouts found in open-pit mining exports:

**Standard schema** (time components split out):
  truck_id, loading_time_min, hauling_time_min, dumping_time_min,
  return_time_min, [queue_time_min], [load_tonnes | payload_tonnes],
  [distance_km | haul_distance_km]

**Timestamp schema** (ISO-8601 start/end per segment):
  truck_id, trip_id, timestamp_start, timestamp_end,
  load_time_min, haul_time_min, dump_time_min, return_time_min,
  payload_tonnes, haul_distance_km, [material_type], [pit_name]

The :meth:`HaulTruckAnalyzer.preprocess` method normalises both layouts to
the standard internal representation before enrichment.

Author: github.com/achmadnaufal
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_TIME_COLUMNS: List[str] = [
    "loading_time_min",
    "hauling_time_min",
    "dumping_time_min",
    "return_time_min",
]

# Column aliases accepted on input and normalised to REQUIRED_TIME_COLUMNS
_TIME_COLUMN_ALIASES: Dict[str, str] = {
    "load_time_min": "loading_time_min",
    "haul_time_min": "hauling_time_min",
    "dump_time_min": "dumping_time_min",
}

# Payload column aliases (all normalised to ``load_tonnes`` internally)
_PAYLOAD_ALIASES: Tuple[str, ...] = ("payload_tonnes", "load_tonnes")

# Distance column aliases (all normalised to ``distance_km`` internally)
_DISTANCE_ALIASES: Tuple[str, ...] = ("haul_distance_km", "distance_km")

OPTIONAL_TIME_COLUMNS: List[str] = ["queue_time_min"]

MINUTES_PER_HOUR: float = 60.0

# Minimum reasonable payload for a haul truck cycle (tonnes).
# Cycles with zero payload are flagged but not rejected outright.
_MIN_PAYLOAD_WARN: float = 0.0

# Maximum plausible single-cycle time (minutes). Anything beyond this is
# likely a data entry error and will be logged as a warning.
_MAX_CYCLE_TIME_WARN: float = 480.0  # 8-hour cap


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_no_duplicate_truck_trip(df: pd.DataFrame) -> None:
    """Raise ValueError when (truck_id, trip_id) combinations are duplicated.

    Parameters
    ----------
    df:
        DataFrame that must contain both ``truck_id`` and ``trip_id`` columns.

    Raises
    ------
    ValueError
        When duplicate (truck_id, trip_id) pairs are detected.
    """
    if "truck_id" not in df.columns or "trip_id" not in df.columns:
        return
    duplicates = df[df.duplicated(subset=["truck_id", "trip_id"], keep=False)]
    if not duplicates.empty:
        pairs = (
            duplicates[["truck_id", "trip_id"]]
            .drop_duplicates()
            .values
            .tolist()
        )
        raise ValueError(
            f"Duplicate (truck_id, trip_id) combinations detected: {pairs}. "
            "Each trip for a given truck must be unique."
        )


def _validate_timestamps_ordered(df: pd.DataFrame) -> None:
    """Raise ValueError when timestamp_end precedes timestamp_start.

    Parameters
    ----------
    df:
        DataFrame that may optionally contain ``timestamp_start`` and
        ``timestamp_end`` columns in ISO-8601 or datetime format.

    Raises
    ------
    ValueError
        When any row has ``timestamp_end <= timestamp_start``.
    """
    if "timestamp_start" not in df.columns or "timestamp_end" not in df.columns:
        return

    starts = pd.to_datetime(df["timestamp_start"], errors="coerce")
    ends = pd.to_datetime(df["timestamp_end"], errors="coerce")

    valid_mask = starts.notna() & ends.notna()
    out_of_order = valid_mask & (ends <= starts)

    if out_of_order.any():
        bad_rows = df.index[out_of_order].tolist()
        raise ValueError(
            f"timestamp_end is not after timestamp_start for row indices: {bad_rows}. "
            "Ensure timestamps are in chronological order."
        )


def _validate_no_missing_truck_ids(df: pd.DataFrame) -> None:
    """Raise ValueError when any ``truck_id`` value is null or blank.

    Parameters
    ----------
    df:
        DataFrame that must contain a ``truck_id`` column.

    Raises
    ------
    ValueError
        When null or empty truck IDs are found.
    """
    if "truck_id" not in df.columns:
        return
    null_mask = df["truck_id"].isnull() | (
        df["truck_id"].astype(str).str.strip() == ""
    )
    if null_mask.any():
        bad_indices = df.index[null_mask].tolist()
        raise ValueError(
            f"Missing or blank truck_id at row indices: {bad_indices}. "
            "Every cycle record must have a valid truck identifier."
        )


def _validate_payload_non_negative(df: pd.DataFrame) -> None:
    """Raise ValueError when any payload column contains negative values.

    Checks both ``load_tonnes`` and ``payload_tonnes`` if present.

    Parameters
    ----------
    df:
        DataFrame to inspect.

    Raises
    ------
    ValueError
        When strictly negative payload values are found.
    """
    for col in _PAYLOAD_ALIASES:
        if col not in df.columns:
            continue
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        neg_mask = numeric_vals.notna() & (numeric_vals < 0)
        if neg_mask.any():
            bad_indices = df.index[neg_mask].tolist()
            raise ValueError(
                f"Negative payload values found in column '{col}' at row indices: "
                f"{bad_indices}. Payload must be >= 0."
            )


def _validate_cycle_times_non_negative(df: pd.DataFrame) -> None:
    """Raise ValueError when required time columns contain negative values.

    Note: :meth:`HaulTruckAnalyzer.preprocess` subsequently clamps negatives
    to zero, but this validation surfaces the data quality issue explicitly
    before clamping occurs.

    Parameters
    ----------
    df:
        DataFrame with normalised snake_case column names.

    Raises
    ------
    ValueError
        When any required time column has at least one negative value.
    """
    all_time_cols = REQUIRED_TIME_COLUMNS + OPTIONAL_TIME_COLUMNS
    # Also check aliases that have not yet been renamed
    alias_cols = list(_TIME_COLUMN_ALIASES.keys())
    cols_to_check = [
        c for c in all_time_cols + alias_cols if c in df.columns
    ]
    for col in cols_to_check:
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        neg_mask = numeric_vals.notna() & (numeric_vals < 0)
        if neg_mask.any():
            bad_indices = df.index[neg_mask].tolist()
            raise ValueError(
                f"Negative cycle time values found in column '{col}' at row "
                f"indices: {bad_indices}. All time components must be >= 0."
            )


# ---------------------------------------------------------------------------
# Pure helper functions (immutable, no side effects)
# ---------------------------------------------------------------------------


def compute_cycle_time(row: pd.Series) -> float:
    """Return total cycle time in minutes by summing all time component columns.

    The function inspects the row for all columns listed in
    ``REQUIRED_TIME_COLUMNS`` and ``OPTIONAL_TIME_COLUMNS``, treating absent
    or null values as zero. Negative component values are clamped to zero so
    that a single bad reading does not produce an impossible total.

    Parameters
    ----------
    row:
        A pandas Series representing a single cycle record. Expected keys are
        the columns in ``REQUIRED_TIME_COLUMNS`` plus any columns from
        ``OPTIONAL_TIME_COLUMNS`` that may be present.

    Returns
    -------
    float
        Sum of all time components rounded to four decimal places. Returns
        ``0.0`` when all components are missing or zero.

    Examples
    --------
    >>> row = pd.Series({
    ...     "loading_time_min": 8.0,
    ...     "hauling_time_min": 18.5,
    ...     "dumping_time_min": 3.0,
    ...     "return_time_min": 16.0,
    ... })
    >>> compute_cycle_time(row)
    45.5
    """
    all_time_cols = REQUIRED_TIME_COLUMNS + OPTIONAL_TIME_COLUMNS
    present_cols = [c for c in all_time_cols if c in row.index]
    if not present_cols:
        return 0.0

    def _safe_float(value: object) -> float:
        """Convert value to float, treating None and NaN as 0.0."""
        try:
            result = float(value)  # type: ignore[arg-type]
            return 0.0 if (result != result) else result  # NaN check via self-inequality
        except (TypeError, ValueError):
            return 0.0

    total = sum(max(_safe_float(row.get(c, 0.0)), 0.0) for c in present_cols)
    return round(total, 4)


def compute_productivity(load_tonnes: float, total_cycle_min: float) -> float:
    """Compute truck productivity in tonnes per hour.

    Productivity is defined as payload divided by cycle time, expressed in
    tonnes per hour. Returns ``0.0`` for any degenerate input that would
    otherwise cause a division-by-zero or a physically meaningless result.

    Parameters
    ----------
    load_tonnes:
        Payload for the cycle in metric tonnes. Must be >= 0.
    total_cycle_min:
        Total cycle duration in minutes. Must be > 0 for a non-zero result.

    Returns
    -------
    float
        Productivity in tonnes / hour, rounded to four decimal places.
        Returns ``0.0`` when ``total_cycle_min`` is zero or negative, or when
        ``load_tonnes`` is negative.

    Examples
    --------
    >>> compute_productivity(220.0, 50.0)
    264.0
    >>> compute_productivity(220.0, 0.0)
    0.0
    """
    if total_cycle_min <= 0.0 or load_tonnes < 0.0:
        return 0.0
    return round((load_tonnes / total_cycle_min) * MINUTES_PER_HOUR, 4)


def identify_bottleneck(row: pd.Series) -> str:
    """Identify the dominant time component (bottleneck) for a single cycle.

    The bottleneck is the time phase that consumes the most time, and therefore
    represents the primary target for operational improvement.

    Parameters
    ----------
    row:
        A pandas Series for a single cycle. Inspects the four required time
        columns (``loading_time_min``, ``hauling_time_min``,
        ``dumping_time_min``, ``return_time_min``) plus ``queue_time_min``
        when available. Absent or null values default to zero.

    Returns
    -------
    str
        Column name with the highest time value (e.g. ``"hauling_time_min"``),
        or ``"unknown"`` when no time columns are present or all values are
        zero or missing.

    Examples
    --------
    >>> row = pd.Series({
    ...     "loading_time_min": 8.0,
    ...     "hauling_time_min": 22.0,
    ...     "dumping_time_min": 3.0,
    ...     "return_time_min": 18.0,
    ... })
    >>> identify_bottleneck(row)
    'hauling_time_min'
    """
    candidates = {c: max(float(row.get(c, 0.0) or 0.0), 0.0)
                  for c in REQUIRED_TIME_COLUMNS + OPTIONAL_TIME_COLUMNS
                  if c in row.index}
    if not candidates or max(candidates.values()) == 0.0:
        return "unknown"
    return max(candidates, key=lambda k: candidates[k])


def compute_utilization(
    active_time_min: float,
    total_available_time_min: float,
) -> float:
    """Compute utilization rate as a percentage.

    Utilization measures the fraction of available shift time that the truck
    spends actively completing cycles. Values are capped at 100.0 % to handle
    minor data discrepancies where recorded active time slightly exceeds the
    nominal shift window.

    Parameters
    ----------
    active_time_min:
        Time the truck was actively in a cycle (minutes). Should be >= 0.
    total_available_time_min:
        Total available shift time (minutes). Must be > 0 for a meaningful
        result.

    Returns
    -------
    float
        Utilization percentage in the range [0.0, 100.0], rounded to four
        decimal places. Returns ``0.0`` when ``total_available_time_min`` is
        zero or negative.

    Examples
    --------
    >>> compute_utilization(600.0, 720.0)
    83.3333
    >>> compute_utilization(720.0, 720.0)
    100.0
    """
    if total_available_time_min <= 0.0 or active_time_min < 0.0:
        return 0.0
    utilization = min((active_time_min / total_available_time_min) * 100.0, 100.0)
    return round(utilization, 4)


def fleet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-truck aggregated statistics from a cleaned cycle DataFrame.

    Aggregates cycle count, average cycle time, total payload, and average
    productivity for each unique truck in the dataset.

    Parameters
    ----------
    df:
        Preprocessed and enriched cycle DataFrame. Must contain at minimum
        ``truck_id`` and ``total_cycle_min`` or ``computed_cycle_min``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame indexed by ``truck_id`` with columns:

        - ``cycle_count`` (int): Number of cycles completed.
        - ``avg_cycle_min`` (float): Mean cycle time in minutes.
        - ``total_tonnes`` (float): Total payload moved in metric tonnes.
        - ``avg_productivity_tph`` (float): Mean productivity in t/h.

        Returns an empty DataFrame when ``df`` is empty or lacks
        ``truck_id``.

    Examples
    --------
    >>> summary = fleet_summary(enriched_df)
    >>> print(summary.loc["TRK01", "cycle_count"])
    5
    """
    if df.empty or "truck_id" not in df.columns:
        return pd.DataFrame()

    agg: Dict[str, Any] = {}

    if "total_cycle_min" in df.columns:
        agg["total_cycle_min"] = ["count", "mean"]
    elif "computed_cycle_min" in df.columns:
        agg["computed_cycle_min"] = ["count", "mean"]

    if "load_tonnes" in df.columns:
        agg["load_tonnes"] = "sum"
    if "productivity_tph" in df.columns:
        agg["productivity_tph"] = "mean"

    if not agg:
        return pd.DataFrame({"truck_id": df["truck_id"].unique()}).set_index("truck_id")

    grouped = df.groupby("truck_id").agg(agg)

    # Flatten multi-level columns produced by multiple aggregation functions
    grouped.columns = [
        "_".join(filter(None, col)).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    rename_map = {
        "total_cycle_min_count": "cycle_count",
        "total_cycle_min_mean": "avg_cycle_min",
        "computed_cycle_min_count": "cycle_count",
        "computed_cycle_min_mean": "avg_cycle_min",
        "load_tonnes_sum": "total_tonnes",
        "productivity_tph_mean": "avg_productivity_tph",
    }
    grouped = grouped.rename(columns=rename_map)
    return grouped.round(3)


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------


class HaulTruckAnalyzer:
    """Mining haul truck cycle time analyzer.

    Provides a full pipeline from raw CSV/Excel data ingestion through
    validation, preprocessing, KPI computation, bottleneck detection, and
    fleet-level summarisation.

    The analyzer is designed to be stateless with respect to data: every
    method that transforms a DataFrame returns a **new** object without
    mutating the input, following immutable data-flow principles.

    Parameters
    ----------
    config:
        Optional configuration dictionary. Supported keys:

        - ``shift_duration_min`` (float): Available shift time in minutes used
          for utilization calculations. Defaults to ``720`` (12-hour shift).
        - ``strict_validation`` (bool): When ``True`` (default), negative cycle
          times raise ``ValueError``; when ``False``, they are silently clamped.

    Attributes
    ----------
    DEFAULT_SHIFT_DURATION_MIN : float
        Class-level default for a 12-hour shift (720 minutes).

    Examples
    --------
    >>> analyzer = HaulTruckAnalyzer()
    >>> df = analyzer.load_data("demo/sample_data.csv")
    >>> result = analyzer.analyze(df)
    >>> print(result["total_records"])
    20
    """

    DEFAULT_SHIFT_DURATION_MIN: float = 720.0  # 12-hour shift

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the analyzer with an optional configuration mapping.

        Parameters
        ----------
        config:
            Optional dictionary of configuration values. Missing keys fall back
            to class-level defaults.
        """
        self.config: Dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load cycle data from a CSV or Excel file.

        The file is read as-is; no schema normalisation is applied at this
        stage. Call :meth:`preprocess` to standardise column names and types
        before further analysis.

        Parameters
        ----------
        filepath:
            Absolute or relative path to the data file. Supported formats:
            ``.csv``, ``.xlsx``, ``.xls``.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame loaded from the file. Row count and column set
            reflect the file contents exactly.

        Raises
        ------
        FileNotFoundError
            When the file does not exist at the specified path.
        ValueError
            When the file extension is not ``.csv``, ``.xlsx``, or ``.xls``.

        Examples
        --------
        >>> df = analyzer.load_data("demo/sample_data.csv")
        >>> df.shape
        (20, 14)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        if path.suffix in (".xlsx", ".xls"):
            return pd.read_excel(filepath)
        if path.suffix == ".csv":
            return pd.read_csv(filepath)
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Use .csv, .xlsx, or .xls."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate the structure and contents of the input DataFrame.

        Performs the following checks in order:

        1. DataFrame must not be empty.
        2. At least the four required time columns must be present (aliases
           such as ``load_time_min`` for ``loading_time_min`` are accepted).
        3. ``truck_id``, when present, must have no null or blank values.
        4. ``timestamp_start`` / ``timestamp_end``, when present, must be in
           chronological order (end strictly after start).
        5. Payload columns must not contain negative values.
        6. Time columns must not contain negative values (unless
           ``strict_validation=False`` in ``config``).

        Parameters
        ----------
        df:
            DataFrame to validate. Column names are normalised to snake_case
            internally for the duration of the check.

        Returns
        -------
        bool
            ``True`` when all checks pass.

        Raises
        ------
        ValueError
            Descriptive error for the first failing validation rule. The
            message always includes the problematic row indices or column
            names to aid data investigation.

        Examples
        --------
        >>> analyzer.validate(df)
        True
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Normalise column names for checking (non-destructive)
        normalised_cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
        normalised_df = df.copy()
        normalised_df.columns = normalised_cols

        # Check required time columns, accepting known aliases
        effective_cols = set(normalised_cols)
        for alias, canonical in _TIME_COLUMN_ALIASES.items():
            if alias in effective_cols:
                effective_cols.add(canonical)

        missing = [c for c in REQUIRED_TIME_COLUMNS if c not in effective_cols]
        if missing:
            raise ValueError(
                f"Missing required time columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # Structural integrity checks
        _validate_no_missing_truck_ids(normalised_df)
        _validate_timestamps_ordered(normalised_df)
        _validate_payload_non_negative(normalised_df)

        strict = self.config.get("strict_validation", True)
        if strict:
            _validate_cycle_times_non_negative(normalised_df)

        return True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise input data without mutating the original.

        Preprocessing steps applied in order:

        1. Drop fully empty rows.
        2. Standardise column names to ``snake_case`` (lower, spaces to ``_``).
        3. Rename column aliases to canonical names (e.g. ``load_time_min``
           becomes ``loading_time_min``, ``payload_tonnes`` becomes
           ``load_tonnes``, ``haul_distance_km`` becomes ``distance_km``).
        4. Clamp negative time values to ``0.0``.
        5. Clamp negative distances to ``0.0``.
        6. Clamp negative payload values to ``0.0``.

        The input DataFrame is **never** mutated; all transformations are
        applied to a working copy.

        Parameters
        ----------
        df:
            Raw input DataFrame as returned by :meth:`load_data` or
            constructed manually.

        Returns
        -------
        pd.DataFrame
            A new, cleaned DataFrame with standardised column names and
            non-negative numeric fields.

        Examples
        --------
        >>> raw = pd.read_csv("demo/sample_data.csv")
        >>> clean = analyzer.preprocess(raw)
        >>> "loading_time_min" in clean.columns
        True
        """
        cleaned = df.dropna(how="all").copy()
        cleaned.columns = [
            c.lower().strip().replace(" ", "_") for c in cleaned.columns
        ]

        # Apply column aliases so both schema variants work uniformly
        cleaned = cleaned.rename(columns=_TIME_COLUMN_ALIASES)

        # Normalise payload column to load_tonnes
        for alias in _PAYLOAD_ALIASES:
            if alias in cleaned.columns and alias != "load_tonnes":
                cleaned = cleaned.rename(columns={alias: "load_tonnes"})
                break

        # Normalise distance column to distance_km
        for alias in _DISTANCE_ALIASES:
            if alias in cleaned.columns and alias != "distance_km":
                cleaned = cleaned.rename(columns={alias: "distance_km"})
                break

        # Clamp negative time values to 0
        time_cols = [
            c for c in cleaned.columns
            if c.endswith("_time_min") or c.endswith("_min")
        ]
        for col in time_cols:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned = cleaned.assign(**{col: cleaned[col].clip(lower=0.0)})

        # Clamp negative distances to 0
        if "distance_km" in cleaned.columns and pd.api.types.is_numeric_dtype(
            cleaned["distance_km"]
        ):
            cleaned = cleaned.assign(distance_km=cleaned["distance_km"].clip(lower=0.0))

        # Clamp negative payloads to 0
        if "load_tonnes" in cleaned.columns and pd.api.types.is_numeric_dtype(
            cleaned["load_tonnes"]
        ):
            cleaned = cleaned.assign(load_tonnes=cleaned["load_tonnes"].clip(lower=0.0))

        return cleaned

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed KPI columns to the preprocessed DataFrame.

        All new columns are appended to a copy of the input; the original is
        not modified.

        Computed columns added:

        - ``computed_cycle_min``: Sum of all recognised time components.
        - ``productivity_tph``: Tonnes per hour based on payload and cycle time.
          Uses ``total_cycle_min`` when available, otherwise falls back to
          ``computed_cycle_min``.
        - ``bottleneck``: Name of the dominant time phase for the cycle.

        Parameters
        ----------
        df:
            Preprocessed DataFrame from :meth:`preprocess`. Column names must
            be in snake_case canonical form.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the three KPI columns appended. The row count
            and index are preserved from the input.

        Examples
        --------
        >>> enriched = analyzer.enrich(preprocessed)
        >>> enriched["bottleneck"].value_counts()
        hauling_time_min    20
        dtype: int64
        """
        enriched = df.copy()

        enriched = enriched.assign(
            computed_cycle_min=enriched.apply(compute_cycle_time, axis=1)
        )

        if "load_tonnes" in enriched.columns:
            cycle_col = (
                "total_cycle_min"
                if "total_cycle_min" in enriched.columns
                else "computed_cycle_min"
            )
            enriched = enriched.assign(
                productivity_tph=enriched.apply(
                    lambda row: compute_productivity(
                        float(row.get("load_tonnes", 0.0) or 0.0),
                        float(row.get(cycle_col, 0.0) or 0.0),
                    ),
                    axis=1,
                )
            )

        enriched = enriched.assign(
            bottleneck=enriched.apply(identify_bottleneck, axis=1)
        )

        return enriched

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the full analysis pipeline and return a summary metrics dict.

        Pipeline: preprocess -> enrich -> aggregate.

        This method does **not** call :meth:`validate`; call it explicitly
        before ``analyze`` when strict data-quality enforcement is required.
        The :meth:`run` convenience method performs both validation and
        analysis in a single call.

        Parameters
        ----------
        df:
            Raw or lightly preprocessed DataFrame. The method calls
            :meth:`preprocess` and :meth:`enrich` internally; passing an
            already-enriched DataFrame is safe but may produce duplicate
            computed columns.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``total_records`` (int): Number of cycles analysed.
            - ``columns`` (list[str]): Column names after preprocessing.
            - ``missing_pct`` (dict[str, float]): Percentage of null values
              per column.
            - ``summary_stats`` (dict): Descriptive statistics for numeric
              columns (from ``pd.DataFrame.describe``).
            - ``totals`` (dict[str, float]): Column sums for numeric fields.
            - ``means`` (dict[str, float]): Column means for numeric fields.
            - ``bottleneck_distribution`` (dict[str, int]): Frequency count
              of each identified bottleneck phase.
            - ``fleet_summary`` (dict): Per-truck aggregated KPIs as produced
              by :func:`fleet_summary`.

        Examples
        --------
        >>> result = analyzer.analyze(df)
        >>> result["total_records"]
        20
        >>> result["bottleneck_distribution"]
        {'hauling_time_min': 20}
        """
        preprocessed = self.preprocess(df)
        enriched = self.enrich(preprocessed)

        result: Dict[str, Any] = {
            "total_records": len(enriched),
            "columns": list(enriched.columns),
            "missing_pct": (
                enriched.isnull().sum() / max(len(enriched), 1) * 100
            ).round(1).to_dict(),
        }

        numeric_df = enriched.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()

        if "bottleneck" in enriched.columns:
            result["bottleneck_distribution"] = (
                enriched["bottleneck"].value_counts().to_dict()
            )

        summary_df = fleet_summary(enriched)
        if not summary_df.empty:
            result["fleet_summary"] = summary_df.to_dict()

        return result

    # ------------------------------------------------------------------
    # Convenience pipeline
    # ------------------------------------------------------------------

    def run(self, filepath: str) -> Dict[str, Any]:
        """Full pipeline: load -> validate -> analyze.

        Combines :meth:`load_data`, :meth:`validate`, and :meth:`analyze`
        into a single call for the common case where data comes from a file
        and all checks should be enforced before analysis.

        Parameters
        ----------
        filepath:
            Path to a ``.csv`` or ``.xlsx``/``.xls`` data file.

        Returns
        -------
        dict
            Analysis result as returned by :meth:`analyze`.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        ValueError
            When validation fails (see :meth:`validate` for details).

        Examples
        --------
        >>> result = analyzer.run("demo/sample_data.csv")
        >>> result["total_records"]
        20
        """
        df = self.load_data(filepath)
        self.validate(df)
        return self.analyze(df)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Convert a flat analysis result dictionary to a tidy DataFrame.

        Nested dictionaries are flattened with dot-separated keys, e.g.
        the key ``"means"`` containing ``{"total_cycle_min": 52.1}`` becomes
        the row ``metric="means.total_cycle_min"``, ``value=52.1``.

        Parameters
        ----------
        result:
            Dictionary as returned by :meth:`analyze`.

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame with ``metric`` (str) and ``value`` (any)
            columns. One row per scalar metric or nested sub-key.

        Examples
        --------
        >>> exported = analyzer.to_dataframe(result)
        >>> exported[exported["metric"] == "total_records"]["value"].iloc[0]
        20
        """
        rows: List[Dict[str, Any]] = []
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append({"metric": f"{key}.{sub_key}", "value": sub_value})
            else:
                rows.append({"metric": key, "value": value})
        return pd.DataFrame(rows)
