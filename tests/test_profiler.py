"""Integration tests for the Profiler class (profiler.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_profiler.config import ColumnChecks, ColumnConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_df(profiler, df, partition="test"):
    return profiler.score(partition_key=partition, df=df, time_column="time")


# ── Train / score round-trip ───────────────────────────────────────────────────

class TestTrainScore:
    def test_train_returns_results_for_all_automl_columns(
        self, make_profiler, synthetic_df
    ):
        profiler = make_profiler()
        results = profiler.train("amsterdam", synthetic_df)
        assert len(results) == 2  # temperature_2m + surface_pressure
        assert all(not r.skipped for r in results)

    def test_score_returns_long_format_dataframe(
        self, make_profiler, synthetic_df
    ):
        profiler = make_profiler()
        profiler.train("amsterdam", synthetic_df)
        preds = profiler.score("amsterdam", synthetic_df)
        assert set(["column", "automl_flag", "automl_score", "quality_violation"]).issubset(
            preds.columns
        )
        # one row per (timestamp × column)
        assert len(preds) == len(synthetic_df) * 2

    def test_score_without_model_returns_none_flags(
        self, make_profiler, synthetic_df
    ):
        """Score before training: model_available=False, flags are None."""
        profiler = make_profiler()  # no training called
        preds = profiler.score("amsterdam", synthetic_df)
        assert preds["automl_flag"].isna().all()
        assert preds["model_available"].eq(False).all()

    def test_each_partition_is_isolated(self, make_profiler, synthetic_df):
        """Models trained for 'london' don't appear under 'amsterdam'."""
        profiler = make_profiler()
        profiler.train("london", synthetic_df)
        status = profiler.model_status("amsterdam")
        assert all(not trained for trained in status.values())

    def test_model_status_after_training(self, make_profiler, synthetic_df):
        profiler = make_profiler()
        profiler.train("amsterdam", synthetic_df)
        status = profiler.model_status("amsterdam")
        assert all(trained for trained in status.values())


# ── Window slicing ─────────────────────────────────────────────────────────────

class TestWindowSlicing:
    def test_window_limits_training_rows(self, make_profiler, synthetic_df):
        """A window smaller than the full DataFrame forces training on fewer rows."""
        small_window = 80
        profiler = make_profiler(window_size=small_window)
        results = profiler.train("amsterdam", synthetic_df)
        for r in results:
            assert not r.skipped
            assert r.n_train_rows <= small_window

    def test_no_window_uses_full_dataframe(self, make_profiler, synthetic_df):
        profiler = make_profiler(window_size=None)
        results = profiler.train("amsterdam", synthetic_df)
        for r in results:
            assert not r.skipped
            assert r.n_train_rows == len(synthetic_df)

    def test_window_larger_than_df_uses_full_df(self, make_profiler, synthetic_df):
        profiler = make_profiler(window_size=10_000)
        results = profiler.train("amsterdam", synthetic_df)
        for r in results:
            assert r.n_train_rows == len(synthetic_df)


# ── Flag threshold override ────────────────────────────────────────────────────

class TestFlagThreshold:
    def _make_profiler_with_threshold(self, tmp_path, threshold):
        from adaptive_profiler import Profiler
        from adaptive_profiler.config import (
            ModelStoreConfig, ProfilerConfig, TrainingConfig
        )
        from adaptive_profiler.storage import LocalStore

        col = ColumnConfig(
            name="temperature_2m",
            automl=True,
            flag_threshold=threshold,
            checks=ColumnChecks(not_null=True),
        )
        config = ProfilerConfig(
            columns=(col,),
            model_store=ModelStoreConfig(backend="local", local_dir=str(tmp_path)),
            training=TrainingConfig(n_trials=2, min_train_rows=10),
        )
        return Profiler(config, LocalStore(tmp_path))

    def test_very_high_threshold_flags_nothing(self, tmp_path, synthetic_df):
        profiler = self._make_profiler_with_threshold(tmp_path, threshold=1e9)
        profiler.train("amsterdam", synthetic_df)
        preds = profiler.score("amsterdam", synthetic_df)
        assert preds["automl_flag"].eq(0).all()

    def test_very_low_threshold_flags_everything(self, tmp_path, synthetic_df):
        profiler = self._make_profiler_with_threshold(tmp_path, threshold=-1e9)
        profiler.train("amsterdam", synthetic_df)
        preds = profiler.score("amsterdam", synthetic_df)
        assert preds["automl_flag"].eq(1).all()

    def test_threshold_overrides_pyod_default(self, tmp_path, synthetic_df):
        """Flags from threshold differ from PyOD's contamination-based flags when threshold changes."""
        p_low = self._make_profiler_with_threshold(tmp_path / "low", threshold=-1e9)
        p_low.train("amsterdam", synthetic_df)
        flags_low = p_low.score("amsterdam", synthetic_df)["automl_flag"].sum()

        p_high = self._make_profiler_with_threshold(tmp_path / "high", threshold=1e9)
        p_high.train("amsterdam", synthetic_df)
        flags_high = p_high.score("amsterdam", synthetic_df)["automl_flag"].sum()

        assert flags_low > flags_high


# ── Manual model override through Profiler ─────────────────────────────────────

class TestManualModelThroughProfiler:
    def test_hbos_override_trains_and_scores(self, tmp_path, synthetic_df):
        from adaptive_profiler import Profiler
        from adaptive_profiler.config import (
            ModelStoreConfig, ProfilerConfig, TrainingConfig
        )
        from adaptive_profiler.storage import LocalStore

        col = ColumnConfig(
            name="temperature_2m",
            automl=True,
            model="HBOS",
            hyperparameters={"n_bins": 10, "contamination": 0.05},
            checks=ColumnChecks(),
        )
        config = ProfilerConfig(
            columns=(col,),
            model_store=ModelStoreConfig(backend="local", local_dir=str(tmp_path)),
            training=TrainingConfig(n_trials=2, min_train_rows=10),
        )
        profiler = Profiler(config, LocalStore(tmp_path))
        results = profiler.train("city", synthetic_df)
        assert not results[0].skipped
        assert results[0].model_name == "HBOS"
        assert results[0].n_trials == 0  # no Optuna


# ── Rule-based quality checks ──────────────────────────────────────────────────

class TestCheckQuality:
    def test_check_quality_on_clean_data_returns_empty(
        self, make_profiler, synthetic_df
    ):
        profiler = make_profiler()
        result = profiler.check_quality(synthetic_df)
        # synthetic_df values are in-range and not null → no violations
        assert result.empty

    def test_check_quality_detects_nulls(self, make_profiler):
        profiler = make_profiler(
            columns=[
                ColumnConfig(
                    name="temperature_2m",
                    automl=False,
                    checks=ColumnChecks(not_null=True),
                )
            ]
        )
        df = pd.DataFrame({"temperature_2m": [10.0, None, 20.0]})
        violations = profiler.check_quality(df)
        assert len(violations) == 1
        assert violations.iloc[0]["rule"] == "null_value"

    def test_quality_summary_returns_counts(self, make_profiler):
        profiler = make_profiler(
            columns=[
                ColumnConfig(
                    name="temperature_2m",
                    automl=False,
                    checks=ColumnChecks(not_null=True),
                )
            ]
        )
        df = pd.DataFrame({"temperature_2m": [None, None, 5.0]})
        summary = profiler.quality_summary(df)
        assert summary.iloc[0]["count"] == 2
