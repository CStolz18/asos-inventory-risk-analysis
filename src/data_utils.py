"""Reusable helpers for data loading and quick profiling."""
from pathlib import Path
import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages."""
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).sort_values(
        "missing_count", ascending=False
    )
