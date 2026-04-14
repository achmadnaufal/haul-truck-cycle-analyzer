"""
Haul truck cycle time analysis, bottleneck identification, and optimization.

This module provides the core HaulTruckAnalyzer class for processing mining
haul truck cycle data, computing productivity KPIs, identifying bottlenecks,
and generating fleet-level summaries.

Author: github.com/achmadnaufal
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_TIME_COLUMNS: List[str] = [
    "loading_time_min",
    "hauling_time_min",
    "dumping_time_min",
    "return_time_min",
]

OPTIONAL_TIME_COLUMNS: List[str] = ["queue_time_min"]

MINUTES_PER_HOUR: float = 60.0


# ---------------------------------------------------------------------------
# Pure helper functions (immutable, no side effects)
# ---------------------------------------------------------------------------


def compute_cycle_time(row: pd.Series) -> float:
    """Return total cycle time in minutes by summing all time component columns.

    Parameters
    ----------
    row:
        A pandas Series representing a single cycle record. Expected keys are
        the columns in ``REQUIRED_TIME_COLUMNS`` plus any columns from
        ``OPTIONAL_TIME_COLUMNS`` that may be present.

    Returns
    -------
    float
        Sum of all time components. Returns 0.0 when all components are
        missing or zero.
    """
    all_time_cols = REQUIRED_TIME_COLUMNS + OPTIONAL_TIME_COLUMNS
    present_cols = [c for c in all_time_cols if c in row.index]
    if not present_cols:
        return 0.0
    total = sum(max(float(row.get(c, 0.0) or 0.0), 0.0) for c in present_cols)
    return round(total, 4)


def compute_productivity(load_tonnes: float, total_cycle_min: float) -> float:
    """Compute truck productivity in tonnes per hour.

    Parameters
    ----------
    load_tonnes:
        Payload for the cycle in metric tonnes.
    total_cycle_min:
        Total cycle duration in minutes.

    Returns
    -------
    float
        Productivity in tonnes / hour. Returns 0.0 when cycle time is zero
        or negative to avoid division by zero.
    """
    if total_cycle_min <= 0.0 or load_tonnes < 0.0:
        return 0.0
    return round((load_tonnes / total_cycle_min) * MINUTES_PER_HOUR, 4)


def identify_bottleneck(row: pd.Series) -> str:
    """Identify the dominant time component (bottleneck) for a single cycle.

    Parameters
    ----------
    row:
        A pandas Series for a single cycle. Inspects the four required time
        columns plus ``queue_time_min`` when available.

    Returns
    -------
    str
        Name of the column with the highest time value, or ``"unknown"`` when
        no time columns are present or all values are zero/missing.
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

    Parameters
    ----------
    active_time_min:
        Time the truck was actively in a cycle (minutes).
    total_available_time_min:
        Total available shift time (minutes).

    Returns
    -------
    float
        Utilization percentage in the range [0.0, 100.0]. Returns 0.0 when
        ``total_available_time_min`` is zero or negative.
    """
    if total_available_time_min <= 0.0 or active_time_min < 0.0:
        return 0.0
    utilization = min((active_time_min / total_available_time_min) * 100.0, 100.0)
    return round(utilization, 4)


def fleet_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-truck aggregated statistics from a cleaned cycle DataFrame.

    Parameters
    ----------
    df:
        Preprocessed cycle DataFrame. Must contain at minimum ``truck_id``
        and ``total_cycle_min``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame indexed by ``truck_id`` with columns:
        ``cycle_count``, ``avg_cycle_min``, ``total_tonnes``,
        ``avg_productivity_tph``.
    """
    if df.empty or "truck_id" not in df.columns:
        return pd.DataFrame()

    agg: Dict[str, Any] = {}

    if "total_cycle_min" in df.columns:
        agg["total_cycle_min"] = ["count", "mean"]
    if "load_tonnes" in df.columns:
        agg["load_tonnes"] = "sum"
    if "productivity_tph" in df.columns:
        agg["productivity_tph"] = "mean"

    if not agg:
        return pd.DataFrame({"truck_id": df["truck_id"].unique()}).set_index("truck_id")

    grouped = df.groupby("truck_id").agg(agg)

    # Flatten multi-level columns
    grouped.columns = [
        "_".join(filter(None, col)).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    rename_map = {
        "total_cycle_min_count": "cycle_count",
        "total_cycle_min_mean": "avg_cycle_min",
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

    Parameters
    ----------
    config:
        Optional configuration dictionary. Supported keys:
        - ``shift_duration_min`` (float): Available shift time used for
          utilization calculations. Defaults to 720 (12-hour shift).

    Examples
    --------
    >>> analyzer = HaulTruckAnalyzer()
    >>> df = analyzer.load_data("demo/sample_data.csv")
    >>> result = analyzer.analyze(df)
    >>> print(result["total_records"])
    20
    """

    DEFAULT_SHIFT_DURATION_MIN: float = 720.0  # 12-hour shift

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialise the analyzer with an optional configuration mapping."""
        self.config: Dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load cycle data from a CSV or Excel file.

        Parameters
        ----------
        filepath:
            Absolute or relative path to the data file. Supported formats:
            ``.csv``, ``.xlsx``, ``.xls``.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame loaded from the file.

        Raises
        ------
        FileNotFoundError
            When the file does not exist at the specified path.
        ValueError
            When the file extension is not supported.
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

        Parameters
        ----------
        df:
            DataFrame to validate.

        Returns
        -------
        bool
            ``True`` when validation passes.

        Raises
        ------
        ValueError
            When the DataFrame is empty or required time columns are absent.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        normalised_cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
        missing = [c for c in REQUIRED_TIME_COLUMNS if c not in normalised_cols]
        if missing:
            raise ValueError(
                f"Missing required time columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        return True

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise input data without mutating the original.

        Steps applied:
        1. Drop fully empty rows.
        2. Standardise column names to snake_case.
        3. Clamp negative time values to zero.
        4. Clamp negative distances to zero.

        Parameters
        ----------
        df:
            Raw input DataFrame.

        Returns
        -------
        pd.DataFrame
            A new, cleaned DataFrame.
        """
        cleaned = df.dropna(how="all").copy()
        cleaned.columns = [
            c.lower().strip().replace(" ", "_") for c in cleaned.columns
        ]

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

        return cleaned

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed KPI columns to the preprocessed DataFrame.

        Computed columns added:
        - ``computed_cycle_min``: Sum of all time components.
        - ``productivity_tph``: Tonnes per hour.
        - ``bottleneck``: Name of the dominant time phase.

        Parameters
        ----------
        df:
            Preprocessed DataFrame from :meth:`preprocess`.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with additional KPI columns appended.
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

        Pipeline: preprocess → enrich → aggregate.

        Parameters
        ----------
        df:
            Raw or lightly preprocessed DataFrame. The method calls
            :meth:`preprocess` and :meth:`enrich` internally.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``total_records`` (int): Number of cycles analysed.
            - ``columns`` (list[str]): Column names after preprocessing.
            - ``missing_pct`` (dict): Percentage of null values per column.
            - ``summary_stats`` (dict): Descriptive statistics for numeric cols.
            - ``totals`` (dict): Column sums for numeric fields.
            - ``means`` (dict): Column means for numeric fields.
            - ``bottleneck_distribution`` (dict): Frequency of each bottleneck.
            - ``fleet_summary`` (dict): Per-truck aggregated KPIs.
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
        """Full pipeline: load → validate → analyze.

        Parameters
        ----------
        filepath:
            Path to a CSV or Excel data file.

        Returns
        -------
        dict
            Analysis result as returned by :meth:`analyze`.
        """
        df = self.load_data(filepath)
        self.validate(df)
        return self.analyze(df)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict) -> pd.DataFrame:
        """Convert a flat analysis result dictionary to a tidy DataFrame.

        Nested dictionaries are flattened with dot-separated keys, e.g.
        ``means.total_cycle_min``.

        Parameters
        ----------
        result:
            Dictionary as returned by :meth:`analyze`.

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame with ``metric`` and ``value`` columns.
        """
        rows = []
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append({"metric": f"{key}.{sub_key}", "value": sub_value})
            else:
                rows.append({"metric": key, "value": value})
        return pd.DataFrame(rows)
