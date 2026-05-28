"""Shared pytest fixtures for adaptive_profiler tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ── Data fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """200-row synthetic weather-like DataFrame with 3 injected anomalies."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature_2m": rng.normal(15.0, 5.0, n),
        "surface_pressure": rng.normal(1013.0, 4.0, n),
        "precipitation": np.clip(rng.exponential(0.4, n), 0.0, None),
        "wind_speed": rng.uniform(0.0, 30.0, n),
    })
    # Inject obvious anomalies so detection tests have signal
    anomaly_idx = [50, 100, 150]
    df.loc[anomaly_idx, "temperature_2m"] += 55.0
    df.loc[anomaly_idx, "surface_pressure"] -= 120.0
    y_true = np.zeros(n, dtype=int)
    y_true[anomaly_idx] = 1
    df["y_true"] = y_true
    return df


# ── Schema / path fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_schema_path() -> Path:
    """Path to the comprehensive sample YAML fixture."""
    return FIXTURE_DIR / "sample_schema.yml"


@pytest.fixture
def minimal_schema_path(tmp_path) -> Path:
    """Write the bare-minimum valid YAML to a temp file and return its path."""
    p = tmp_path / "minimal.yml"
    p.write_text(
        "version: 1\n"
        "model_store:\n"
        "  backend: local\n"
        "  local_dir: /tmp/ap_minimal\n"
        "columns:\n"
        "  - name: temperature_2m\n"
        "    automl: false\n"
    )
    return p


# ── Profiler factory fixture ───────────────────────────────────────────────────

@pytest.fixture
def make_profiler(tmp_path):
    """Factory that returns a Profiler backed by a temp LocalStore.

    Usage::

        def test_something(make_profiler, synthetic_df):
            profiler = make_profiler()
            profiler.train("city_a", synthetic_df)
    """
    from adaptive_profiler import Profiler
    from adaptive_profiler.config import (
        ColumnChecks,
        ColumnConfig,
        ModelStoreConfig,
        ProfilerConfig,
        TrainingConfig,
    )
    from adaptive_profiler.storage import LocalStore

    def _factory(
        columns=None,
        n_trials: int = 2,
        window_size: int | None = None,
        seed: int = 42,
    ) -> Profiler:
        if columns is None:
            columns = [
                ColumnConfig(
                    name="temperature_2m",
                    automl=True,
                    checks=ColumnChecks(not_null=True),
                ),
                ColumnConfig(
                    name="surface_pressure",
                    automl=True,
                    checks=ColumnChecks(not_null=True),
                ),
            ]
        config = ProfilerConfig(
            columns=tuple(columns),
            model_store=ModelStoreConfig(backend="local", local_dir=str(tmp_path)),
            training=TrainingConfig(
                n_trials=n_trials,
                min_train_rows=10,
                window_size=window_size,
                seed=seed,
            ),
        )
        return Profiler(config, LocalStore(tmp_path))

    return _factory
