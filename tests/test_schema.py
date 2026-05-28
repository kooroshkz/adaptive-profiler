"""Tests for YAML configuration parsing (schema.py)."""

from __future__ import annotations

import pytest

from adaptive_profiler.config import (
    ColumnChecks,
    ColumnConfig,
    ModelStoreConfig,
    ProfilerConfig,
    TrainingConfig,
)


# ── Comprehensive YAML round-trip ──────────────────────────────────────────────

class TestSampleYamlParsing:
    """Parse sample_schema.yml and assert every field is read correctly."""

    def test_columns_count(self, sample_schema_path):
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        assert len(cfg.columns) == 4

    def test_model_store(self, sample_schema_path):
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        assert cfg.model_store.backend == "local"
        assert "/tmp" in cfg.model_store.local_dir

    def test_training_n_trials(self, sample_schema_path):
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        assert cfg.training.n_trials == 3

    def test_training_min_train_rows(self, sample_schema_path):
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        assert cfg.training.min_train_rows == 50

    def test_training_window_size(self, sample_schema_path):
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        assert cfg.training.window_size == 180

    def test_training_seed(self, sample_schema_path):
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        assert cfg.training.seed == 42

    def test_standard_automl_column(self, sample_schema_path):
        """temperature_2m: plain automl, range check, no overrides."""
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        col = next(c for c in cfg.columns if c.name == "temperature_2m")
        assert col.automl is True
        assert col.flag_threshold is None
        assert col.model is None
        assert col.hyperparameters == {}
        assert col.checks.not_null is True
        assert col.checks.range == (-60.0, 60.0)

    def test_flag_threshold_column(self, sample_schema_path):
        """surface_pressure: flag_threshold is parsed as float."""
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        col = next(c for c in cfg.columns if c.name == "surface_pressure")
        assert col.automl is True
        assert col.flag_threshold == pytest.approx(0.45)
        assert col.model is None

    def test_manual_model_column(self, sample_schema_path):
        """precipitation: model + hyperparameters override Optuna."""
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        col = next(c for c in cfg.columns if c.name == "precipitation")
        assert col.automl is True
        assert col.model == "HBOS"
        assert col.hyperparameters["n_bins"] == 15
        assert col.hyperparameters["contamination"] == pytest.approx(0.03)

    def test_rules_only_column(self, sample_schema_path):
        """wind_speed: automl disabled, still has rule checks."""
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        col = next(c for c in cfg.columns if c.name == "wind_speed")
        assert col.automl is False
        assert col.checks.not_null is True
        assert col.checks.range == (0.0, 200.0)

    def test_automl_columns_filter(self, sample_schema_path):
        """automl_columns returns only the three automl: true columns."""
        cfg = ProfilerConfig.from_yaml(sample_schema_path)
        names = cfg.automl_column_names
        assert "wind_speed" not in names
        assert len(names) == 3


# ── Defaults when fields are omitted ──────────────────────────────────────────

class TestDefaults:
    def test_window_size_defaults_to_none(self, minimal_schema_path):
        cfg = ProfilerConfig.from_yaml(minimal_schema_path)
        assert cfg.training.window_size is None

    def test_flag_threshold_defaults_to_none(self, minimal_schema_path):
        cfg = ProfilerConfig.from_yaml(minimal_schema_path)
        col = cfg.columns[0]
        assert col.flag_threshold is None

    def test_model_defaults_to_none(self, minimal_schema_path):
        cfg = ProfilerConfig.from_yaml(minimal_schema_path)
        col = cfg.columns[0]
        assert col.model is None

    def test_hyperparameters_defaults_to_empty_dict(self, minimal_schema_path):
        cfg = ProfilerConfig.from_yaml(minimal_schema_path)
        col = cfg.columns[0]
        assert col.hyperparameters == {}

    def test_training_defaults(self, minimal_schema_path):
        cfg = ProfilerConfig.from_yaml(minimal_schema_path)
        assert cfg.training.n_trials == 30
        assert cfg.training.min_train_rows == 100
        assert cfg.training.seed == 42


# ── Validation ─────────────────────────────────────────────────────────────────

def test_unsupported_version_raises(tmp_path):
    p = tmp_path / "bad.yml"
    p.write_text("version: 2\ncolumns: []\n")
    with pytest.raises(ValueError, match="Unsupported schema version"):
        ProfilerConfig.from_yaml(p)


def test_repr(sample_schema_path):
    cfg = ProfilerConfig.from_yaml(sample_schema_path)
    r = repr(cfg)
    assert "ProfilerConfig" in r
    assert "temperature_2m" in r


# ── ColumnChecks.violations ────────────────────────────────────────────────────

class TestColumnChecksViolations:
    def setup_method(self):
        self.checks = ColumnChecks(not_null=True, range=(0.0, 100.0), type="float")

    def test_clean_value(self):
        assert self.checks.violations(42.0) == []

    def test_null_triggers_not_null(self):
        assert self.checks.violations(None) == ["null_value"]

    def test_nan_triggers_not_null(self):
        import math
        assert self.checks.violations(float("nan")) == ["null_value"]

    def test_out_of_range_high(self):
        violations = self.checks.violations(150.0)
        assert any("out_of_range" in v for v in violations)

    def test_out_of_range_low(self):
        violations = self.checks.violations(-5.0)
        assert any("out_of_range" in v for v in violations)

    def test_null_allowed_when_not_null_false(self):
        checks = ColumnChecks(not_null=False, range=(0.0, 100.0))
        assert checks.violations(None) == []

    def test_boundary_values_are_clean(self):
        assert self.checks.violations(0.0) == []
        assert self.checks.violations(100.0) == []


# ── Programmatic construction ──────────────────────────────────────────────────

def test_column_config_construction():
    col = ColumnConfig(
        name="pressure",
        automl=True,
        flag_threshold=0.7,
        model="LOF",
        hyperparameters={"n_neighbors": 20, "contamination": 0.02},
    )
    assert col.flag_threshold == pytest.approx(0.7)
    assert col.model == "LOF"
    assert col.hyperparameters["n_neighbors"] == 20


def test_training_config_window_size():
    cfg = TrainingConfig(n_trials=10, window_size=5000)
    assert cfg.window_size == 5000
    assert cfg.n_trials == 10
