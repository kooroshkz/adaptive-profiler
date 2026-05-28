"""Tests for rule-based quality checks (quality.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from adaptive_profiler.quality import check_dataframe, quality_summary
from adaptive_profiler.config import ColumnChecks, ColumnConfig


def _col(name: str, **kwargs) -> ColumnConfig:
    checks = ColumnChecks(**kwargs)
    return ColumnConfig(name=name, checks=checks)


# ── check_dataframe ────────────────────────────────────────────────────────────

class TestCheckDataframe:
    def test_clean_data_returns_empty(self):
        df = pd.DataFrame({"temp": [10.0, 20.0, 30.0]})
        cols = [_col("temp", not_null=True, range=(0.0, 50.0))]
        result = check_dataframe(df, cols)
        assert result.empty

    def test_null_violation_detected(self):
        df = pd.DataFrame({"temp": [10.0, None, 30.0]})
        cols = [_col("temp", not_null=True)]
        result = check_dataframe(df, cols)
        assert len(result) == 1
        assert result.iloc[0]["rule"] == "null_value"

    def test_range_violation_detected(self):
        df = pd.DataFrame({"temp": [10.0, 200.0, 30.0]})
        cols = [_col("temp", range=(0.0, 100.0))]
        result = check_dataframe(df, cols)
        assert len(result) == 1
        assert "out_of_range" in result.iloc[0]["rule"]

    def test_multiple_violations_across_columns(self):
        df = pd.DataFrame({
            "temp": [None, 20.0],
            "pressure": [1010.0, 5000.0],  # 5000 out of range
        })
        cols = [
            _col("temp", not_null=True),
            _col("pressure", range=(800.0, 1200.0)),
        ]
        result = check_dataframe(df, cols)
        assert len(result) == 2

    def test_missing_column_is_skipped(self):
        df = pd.DataFrame({"temp": [10.0, 20.0]})
        cols = [_col("temp"), _col("pressure", not_null=True)]
        result = check_dataframe(df, cols)
        assert result.empty  # temp is fine; pressure column not in df → skipped

    def test_result_columns(self):
        df = pd.DataFrame({"temp": [None]})
        cols = [_col("temp", not_null=True)]
        result = check_dataframe(df, cols)
        assert set(result.columns) == {"column", "row_index", "rule", "value"}

    def test_value_is_preserved(self):
        df = pd.DataFrame({"temp": [999.0]})
        cols = [_col("temp", range=(0.0, 100.0))]
        result = check_dataframe(df, cols)
        assert result.iloc[0]["value"] == 999.0
        assert result.iloc[0]["column"] == "temp"


# ── quality_summary ────────────────────────────────────────────────────────────

class TestQualitySummary:
    def test_empty_violations_returns_empty(self):
        df = pd.DataFrame({"temp": [10.0]})
        cols = [_col("temp")]
        violations = check_dataframe(df, cols)
        summary = quality_summary(violations)
        assert summary.empty

    def test_counts_per_column_rule(self):
        df = pd.DataFrame({"temp": [None, None, None, 10.0]})
        cols = [_col("temp", not_null=True)]
        violations = check_dataframe(df, cols)
        summary = quality_summary(violations)
        assert len(summary) == 1
        assert summary.iloc[0]["count"] == 3
        assert summary.iloc[0]["rule"] == "null_value"

    def test_summary_columns(self):
        df = pd.DataFrame({"temp": [None]})
        cols = [_col("temp", not_null=True)]
        violations = check_dataframe(df, cols)
        summary = quality_summary(violations)
        assert set(summary.columns) == {"column", "rule", "count"}

    def test_sorted_descending_by_count(self):
        df = pd.DataFrame({
            "a": [None, None, None, 1.0],
            "b": [None, 2.0, 2.0, 2.0],
        })
        cols = [_col("a", not_null=True), _col("b", not_null=True)]
        violations = check_dataframe(df, cols)
        summary = quality_summary(violations)
        counts = summary["count"].tolist()
        assert counts == sorted(counts, reverse=True)
