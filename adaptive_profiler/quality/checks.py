"""Rule-based data quality checks, independent of the anomaly detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config.schema import ColumnConfig


@dataclass
class QualityViolation:
    """A single rule violation for one (column, row) pair."""

    column: str
    row_index: Any      # original DataFrame index value
    rule: str           # e.g. "null_value", "out_of_range([0, 500])", "type_error(expected=float)"
    value: Any

    def __str__(self) -> str:
        return f"{self.column}[{self.row_index}]: {self.rule} (value={self.value!r})"


def check_dataframe(
    df: pd.DataFrame,
    columns: list[ColumnConfig],
) -> pd.DataFrame:
    """Run all schema rules against *df* and return a violations DataFrame.

    Returns
    -------
    DataFrame with columns: ``column``, ``row_index``, ``rule``, ``value``.
    Empty when the data is clean.
    """
    rows: list[dict[str, Any]] = []
    for col_cfg in columns:
        col = col_cfg.name
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            for rule in col_cfg.checks.violations(val):
                rows.append({
                    "column": col,
                    "row_index": idx,
                    "rule": rule,
                    "value": val,
                })
    return pd.DataFrame(rows, columns=["column", "row_index", "rule", "value"])


def quality_summary(violations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate violation counts per column and rule type.

    Parameters
    ----------
    violations: Output of :func:`check_dataframe`.

    Returns
    -------
    DataFrame with columns: ``column``, ``rule``, ``count``, sorted by count desc.
    """
    if violations.empty:
        return pd.DataFrame(columns=["column", "rule", "count"])
    return (
        violations.groupby(["column", "rule"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
