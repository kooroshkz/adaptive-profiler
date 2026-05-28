"""Tests for the training logic (trainer.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_profiler.config import ColumnChecks, ColumnConfig, TrainingConfig
from adaptive_profiler.detection import train_column


def _make_df(n: int = 120, with_labels: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"col": rng.normal(0, 1, n)})
    if with_labels:
        y = np.zeros(n, dtype=int)
        y[:5] = 1  # a handful of positives
        df["y_true"] = y
    return df


def _cfg(
    model: str | None = None,
    hyperparameters: dict | None = None,
) -> ColumnConfig:
    return ColumnConfig(
        name="col",
        automl=True,
        checks=ColumnChecks(),
        model=model,
        hyperparameters=hyperparameters or {},
    )


def _training_cfg(n_trials: int = 2) -> TrainingConfig:
    return TrainingConfig(n_trials=n_trials, min_train_rows=10, seed=42)


# ── Skip conditions ────────────────────────────────────────────────────────────

def test_skip_when_column_not_in_dataframe():
    df = pd.DataFrame({"other": [1.0, 2.0]})
    result, artifact, meta = train_column(df, _cfg(), _training_cfg(), "p1")
    assert result.skipped
    assert "not in DataFrame" in result.skip_reason
    assert artifact is None


def test_skip_when_below_min_train_rows():
    df = _make_df(n=5)
    cfg = _training_cfg()
    # min_train_rows=10, df only has 5 rows
    result, artifact, meta = train_column(df, _cfg(), cfg, "p1")
    assert result.skipped
    assert "min=" in result.skip_reason


# ── Manual model override ──────────────────────────────────────────────────────

class TestManualOverride:
    """When col_cfg.model is set, Optuna is skipped entirely."""

    @pytest.mark.parametrize("model_name", ["LOF", "HBOS", "IForest", "ECOD", "COPOD"])
    def test_supported_models_train_without_optuna(self, model_name):
        df = _make_df(n=100)
        col = _cfg(model=model_name)
        result, artifact, meta = train_column(df, col, _training_cfg(), "p1")
        assert not result.skipped
        assert result.model_name == model_name
        assert result.n_trials == 0  # no Optuna search
        assert artifact is not None
        assert meta["manual_override"] is True

    def test_hyperparameters_are_forwarded(self):
        df = _make_df(n=100)
        col = _cfg(model="LOF", hyperparameters={"n_neighbors": 7, "contamination": 0.04})
        result, artifact, meta = train_column(df, col, _training_cfg(), "p1")
        assert not result.skipped
        assert meta["params"]["n_neighbors"] == 7
        assert meta["params"]["contamination"] == pytest.approx(0.04)

    def test_contamination_default_applied_when_missing(self):
        df = _make_df(n=100)
        col = _cfg(model="ECOD")  # no contamination in hyperparameters
        result, artifact, meta = train_column(df, col, _training_cfg(), "p1")
        assert not result.skipped
        assert meta["params"]["contamination"] == pytest.approx(0.05)

    def test_unsupported_model_is_skipped(self):
        df = _make_df(n=100)
        col = _cfg(model="RandomForest")  # not in SUPPORTED_MODELS
        result, artifact, meta = train_column(df, col, _training_cfg(), "p1")
        assert result.skipped
        assert "unsupported model" in result.skip_reason
        assert artifact is None


# ── Standard Optuna path ───────────────────────────────────────────────────────

def test_standard_training_returns_result():
    df = _make_df(n=120)
    result, artifact, meta = train_column(df, _cfg(), _training_cfg(n_trials=2), "p1")
    assert not result.skipped
    assert result.model_name in ["IForest", "LOF", "HBOS", "COPOD", "ECOD"]
    assert result.n_trials == 2
    assert artifact is not None
    assert "preprocess" in artifact
    assert "model" in artifact


def test_training_with_labels_uses_f2_objective():
    df = _make_df(n=120, with_labels=True)
    result, artifact, meta = train_column(df, _cfg(), _training_cfg(n_trials=2), "p1")
    assert not result.skipped
    assert meta["label_available"] is True


def test_artifact_can_score_data():
    df = _make_df(n=120)
    result, artifact, _ = train_column(df, _cfg(), _training_cfg(n_trials=2), "p1")
    assert not result.skipped
    X = df[["col"]].to_numpy(dtype=float)
    X_proc = artifact["preprocess"].transform(X)
    flags = artifact["model"].predict(X_proc)
    assert len(flags) == len(df)


def test_partition_key_stored_in_result():
    df = _make_df(n=120)
    result, _, _ = train_column(df, _cfg(), _training_cfg(n_trials=1), "city_amsterdam")
    assert result.partition_key == "city_amsterdam"
