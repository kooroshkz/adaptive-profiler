"""Training logic: Optuna HPO over PyOD detectors, with F2 or proxy objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import _build_model, _create_study, _suggest_params
from ..config.schema import ColumnConfig, TrainingConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# Public result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Outcome of training a single column model."""

    partition_key: str
    column: str
    model_name: str
    n_train_rows: int
    best_val_f2: float          # 0.0 when labels were unavailable
    n_trials: int
    label_available: bool = False
    skipped: bool = False
    skip_reason: str = ""

    def __str__(self) -> str:
        if self.skipped:
            return f"[SKIP] {self.partition_key}/{self.column}: {self.skip_reason}"
        label_note = f"val_f2={self.best_val_f2:.3f}" if self.label_available else "proxy_obj"
        return (
            f"[OK]   {self.partition_key}/{self.column}: "
            f"model={self.model_name} {label_note} n_rows={self.n_train_rows}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_preprocess() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F2 = (5·TP) / (5·TP + 4·FN + FP) — weights recall twice as heavily as precision."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = 5 * tp + 4 * fn + fp
    return 5 * tp / denom if denom > 0 else 0.0


def _split(
    X: np.ndarray, y: np.ndarray, ratio: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, counts = np.unique(y, return_counts=True)
    can_stratify = len(labels) > 1 and int(counts.min()) >= 2 and len(X) >= 4
    idx = np.arange(len(X))
    if can_stratify:
        idx_tr, idx_va = train_test_split(idx, test_size=1 - ratio, random_state=42, stratify=y)
    else:
        cut = max(1, min(int(len(X) * ratio), len(X) - 1))
        idx_tr, idx_va = idx[:cut], idx[cut:]
    return X[idx_tr], X[idx_va], y[idx_tr], y[idx_va]


# ──────────────────────────────────────────────────────────────────────────────
# Manual override helper
# ──────────────────────────────────────────────────────────────────────────────

def _train_manual(
    col: str,
    partition_key: str,
    model_name: str,
    hyperparameters: dict[str, Any],
    X: np.ndarray,
) -> tuple["TrainingResult", dict[str, Any] | None, dict[str, Any] | None]:
    """Build and fit a model directly from schema-specified name + hyperparameters.

    Called when ``col_cfg.model`` is set, bypassing the Optuna search entirely.
    The engineer is responsible for supplying valid hyperparameters; any key
    not recognised by the chosen model is silently ignored by ``_build_model``.

    ``contamination`` defaults to 0.05 if omitted from ``hyperparameters``.
    """
    from .models import SUPPORTED_MODELS

    if model_name not in SUPPORTED_MODELS:
        return (
            TrainingResult(
                partition_key=partition_key,
                column=col,
                model_name="",
                n_train_rows=0,
                best_val_f2=0.0,
                n_trials=0,
                skipped=True,
                skip_reason=(
                    f"unsupported model: {model_name!r}. "
                    f"Supported: {SUPPORTED_MODELS}"
                ),
            ),
            None,
            None,
        )

    params: dict[str, Any] = {"contamination": 0.05, **hyperparameters}
    preprocess = _build_preprocess()
    model = _build_model(model_name, params)
    model.fit(preprocess.fit_transform(X))

    result = TrainingResult(
        partition_key=partition_key,
        column=col,
        model_name=model_name,
        n_train_rows=len(X),
        best_val_f2=0.0,
        n_trials=0,
        label_available=False,
    )
    artifact: dict[str, Any] = {"preprocess": preprocess, "model": model}
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "params": params,
        "n_train_rows": len(X),
        "n_trials": 0,
        "best_objective_value": 0.0,
        "label_available": False,
        "manual_override": True,
    }
    return result, artifact, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Public
# ──────────────────────────────────────────────────────────────────────────────

def train_column(
    df: pd.DataFrame,
    col_cfg: ColumnConfig,
    training_cfg: TrainingConfig,
    partition_key: str,
) -> tuple[TrainingResult, dict[str, Any] | None, dict[str, Any] | None]:
    """Train one anomaly detector for a single column.

    Objective strategy
    ------------------
    - When ``y_true`` is present in *df*: F2 on a stratified validation split.
      (F2 weights recall 2× — better for catching rare anomalies.)
    - Without labels: standard deviation of anomaly scores on validation data —
      a proxy for detector discriminativeness.

    The final model is refit on the entire dataset so its contamination
    threshold reflects the full distribution.

    Parameters
    ----------
    df:            DataFrame containing at least the target column.
    col_cfg:       Column configuration from the profiling schema.
    training_cfg:  Optuna / min-rows settings from the schema.
    partition_key: Slice identifier (e.g., city name, sensor id).

    Returns
    -------
    ``(TrainingResult, artifact, metadata)`` where *artifact* is
    ``{"preprocess": Pipeline, "model": PyOD_detector}`` ready for storage,
    and both are ``None`` when training was skipped.
    """
    col = col_cfg.name
    _skip = lambda reason: (  # noqa: E731
        TrainingResult(
            partition_key=partition_key, column=col, model_name="",
            n_train_rows=0, best_val_f2=0.0, n_trials=0,
            skipped=True, skip_reason=reason,
        ),
        None, None,
    )

    if col not in df.columns:
        return _skip("column not in DataFrame")

    X = df[[col]].to_numpy(dtype=float)
    n_valid = int(np.isfinite(X).sum())
    if n_valid < training_cfg.min_train_rows:
        return _skip(f"only {n_valid} non-null rows (min={training_cfg.min_train_rows})")

    # ── Manual model override: skip Optuna, use specified model + params ──────
    if col_cfg.model:
        return _train_manual(
            col=col,
            partition_key=partition_key,
            model_name=col_cfg.model,
            hyperparameters=col_cfg.hyperparameters,
            X=X,
        )

    has_labels = "y_true" in df.columns
    y = df["y_true"].to_numpy(dtype=int) if has_labels else np.zeros(len(X), dtype=int)
    X_tr, X_va, y_tr, y_va = _split(X, y)

    study = _create_study(training_cfg.seed)

    def objective(trial: optuna.Trial) -> float:
        model_name, params = _suggest_params(trial)
        pre = _build_preprocess()
        mdl = _build_model(model_name, params)
        mdl.fit(pre.fit_transform(X_tr))
        X_va_p = pre.transform(X_va)
        if has_labels:
            return _f2_score(y_va, mdl.predict(X_va_p).astype(int))
        return float(np.std(mdl.decision_function(X_va_p)))

    study.optimize(objective, n_trials=training_cfg.n_trials, show_progress_bar=False)

    best_model_name: str = study.best_trial.params["model"]
    best_params = {k: v for k, v in study.best_trial.params.items() if k != "model"}

    preprocess = _build_preprocess()
    model = _build_model(best_model_name, best_params)
    model.fit(preprocess.fit_transform(X))

    best_score = float(study.best_value)
    result = TrainingResult(
        partition_key=partition_key,
        column=col,
        model_name=best_model_name,
        n_train_rows=len(X),
        best_val_f2=best_score if has_labels else 0.0,
        n_trials=training_cfg.n_trials,
        label_available=has_labels,
    )
    artifact: dict[str, Any] = {"preprocess": preprocess, "model": model}
    metadata: dict[str, Any] = {
        "model_name": best_model_name,
        "params": best_params,
        "n_train_rows": len(X),
        "n_trials": training_cfg.n_trials,
        "best_objective_value": best_score,
        "label_available": has_labels,
    }
    return result, artifact, metadata
