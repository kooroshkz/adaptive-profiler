"""PyOD model registry and Optuna search-space definitions.

This module is an implementation detail of the library — users interact with
the ``Profiler`` class rather than these functions directly.  The list of
``SUPPORTED_MODELS`` is exposed publicly for inspection.
"""

from __future__ import annotations

from typing import Any

import optuna

# ──────────────────────────────────────────────────────────────────────────────
# Public: model registry
# ──────────────────────────────────────────────────────────────────────────────

#: All PyOD detectors that the AutoML search considers.
#: IForest / HBOS / COPOD / ECOD scale to large datasets (O(n) or O(n log n)).
#: LOF is O(n²) but included because it performs well on small windows.
SUPPORTED_MODELS: list[str] = ["IForest", "LOF", "HBOS", "COPOD", "ECOD"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(model_name: str, params: dict[str, Any]):
    """Instantiate a PyOD detector from a name + params dict."""
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF

    if model_name == "IForest":
        return IForest(
            contamination=params["contamination"],
            n_estimators=params.get("n_estimators", 100),
            max_samples=params.get("max_samples", "auto"),
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "LOF":
        return LOF(
            contamination=params["contamination"],
            n_neighbors=params.get("n_neighbors", 20),
            leaf_size=params.get("leaf_size", 30),
            metric="minkowski",
            p=2,
        )
    if model_name == "HBOS":
        return HBOS(
            contamination=params["contamination"],
            n_bins=params.get("n_bins", 10),
            alpha=params.get("alpha", 0.1),
            tol=params.get("tol", 0.5),
        )
    if model_name == "COPOD":
        return COPOD(contamination=params["contamination"])
    if model_name == "ECOD":
        return ECOD(contamination=params["contamination"])
    raise ValueError(f"Unknown model: {model_name!r}. Supported: {SUPPORTED_MODELS}")


def _suggest_params(trial: optuna.Trial) -> tuple[str, dict[str, Any]]:
    """Propose a model name + hyperparameters for an Optuna trial."""
    model_name = trial.suggest_categorical("model", SUPPORTED_MODELS)
    params: dict[str, Any] = {
        # Upper bound 0.20 lets the search reach realistic contamination rates
        "contamination": trial.suggest_float("contamination", 0.001, 0.20, log=True),
    }
    if model_name == "IForest":
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
        params["max_samples"] = trial.suggest_float("max_samples", 0.3, 1.0)
    elif model_name == "LOF":
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 5, 100)
        params["leaf_size"] = trial.suggest_int("leaf_size", 10, 80)
    elif model_name == "HBOS":
        params["n_bins"] = trial.suggest_int("n_bins", 5, 50)
        params["alpha"] = trial.suggest_float("alpha", 0.01, 0.5)
        params["tol"] = trial.suggest_float("tol", 0.05, 0.9)
    return model_name, params


def _create_study(seed: int) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(direction="maximize", sampler=sampler)
