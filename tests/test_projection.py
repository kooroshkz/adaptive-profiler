"""Tests for the ScalingBenchmark cost-projection tool (projection.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_profiler import ScalingBenchmark


@pytest.fixture
def small_df():
    """Tiny DataFrame sufficient for a quick benchmark sweep."""
    rng = np.random.default_rng(7)
    n = 3_000
    return pd.DataFrame({
        "col_a": rng.normal(0.0, 1.0, n),
        "col_b": rng.normal(5.0, 2.0, n),
        "col_c": rng.exponential(1.0, n),
    })


# ── Construction ───────────────────────────────────────────────────────────────

class TestInit:
    def test_basic_construction(self, small_df):
        bench = ScalingBenchmark(small_df, columns=["col_a", "col_b"])
        assert bench._X.shape == (len(small_df), 2)

    def test_missing_columns_raises(self, small_df):
        with pytest.raises(ValueError, match="None of the requested columns"):
            ScalingBenchmark(small_df, columns=["nonexistent"])

    def test_partial_columns_warns(self, small_df):
        with pytest.warns(UserWarning, match="not found"):
            bench = ScalingBenchmark(small_df, columns=["col_a", "ghost_col"])
        assert bench._columns == ["col_a"]

    def test_repr(self, small_df):
        bench = ScalingBenchmark(small_df, columns=["col_a", "col_b"])
        r = repr(bench)
        assert "ScalingBenchmark" in r
        assert "3,000" in r  # n_rows formatted


# ── Predict before fit raises ──────────────────────────────────────────────────

def test_predict_before_run_raises(small_df):
    bench = ScalingBenchmark(small_df, columns=["col_a"])
    with pytest.raises(RuntimeError, match="Call .run\\(\\) then .fit\\(\\)"):
        bench.predict(n=1000, m=2, k=10)


def test_fit_before_run_raises(small_df):
    bench = ScalingBenchmark(small_df, columns=["col_a"])
    with pytest.raises(RuntimeError, match="Call .run\\(\\)"):
        bench.fit()


def test_report_before_fit_raises(small_df):
    bench = ScalingBenchmark(small_df, columns=["col_a"])
    with pytest.raises(RuntimeError, match="Call .run\\(\\) then .fit\\(\\)"):
        bench.report()


# ── Run → Fit → Predict → Report ──────────────────────────────────────────────

class TestRunFitPredictReport:
    """Uses a very small custom grid so tests finish in a few seconds."""

    @pytest.fixture
    def fitted_bench(self, small_df):
        bench = ScalingBenchmark(small_df, columns=["col_a", "col_b"], seed=0)
        bench.run(
            n_vals=[500, 1_000],
            m_vals=[1, 2],
            k_vals=[5, 25],
            verbose=False,
        )
        bench.fit()
        return bench

    def test_results_dataframe_shape(self, fitted_bench):
        results = fitted_bench.results
        assert results is not None
        # 2 n_vals × 2 m_vals × 2 k_vals = 8 cells
        assert len(results) == 8
        assert set(results.columns) >= {"n", "m", "k", "total_sec"}

    def test_formula_params_keys(self, fitted_bench):
        params = fitted_bench.formula_params
        assert params is not None
        assert set(params.keys()) >= {"alpha", "beta", "delta", "gamma", "r2", "n_obs"}

    def test_alpha_is_positive(self, fitted_bench):
        assert fitted_bench.formula_params["alpha"] > 0

    def test_r2_is_reasonable(self, fitted_bench):
        # power-law should explain variance reasonably well
        assert fitted_bench.formula_params["r2"] > 0.5

    def test_predict_returns_positive_float(self, fitted_bench):
        t = fitted_bench.predict(n=5_000, m=3, k=20)
        assert isinstance(t, float)
        assert t > 0.0

    def test_predict_scales_with_n(self, fitted_bench):
        t_small = fitted_bench.predict(n=1_000, m=2, k=10)
        t_large = fitted_bench.predict(n=10_000, m=2, k=10)
        assert t_large > t_small

    def test_predict_scales_with_k(self, fitted_bench):
        t_few = fitted_bench.predict(n=2_000, m=2, k=5)
        t_many = fitted_bench.predict(n=2_000, m=2, k=25)
        assert t_many > t_few

    def test_report_returns_string(self, fitted_bench):
        report = fitted_bench.report(target_n=2_000, m=2, k=10)
        assert isinstance(report, str)
        assert "α" in report or "alpha" in report.lower()

    def test_report_contains_projections(self, fitted_bench):
        report = fitted_bench.report(target_n=1_000, m=2, k=10, scale_factors=[1, 2, 10])
        assert "2×" in report
        assert "10×" in report

    def test_method_chaining(self, small_df):
        """run() returns self so .run().fit() works."""
        bench = ScalingBenchmark(small_df, columns=["col_a"], seed=0)
        result = bench.run(n_vals=[500], m_vals=[1], k_vals=[3], verbose=False)
        assert result is bench  # same object

    def test_results_is_copy(self, fitted_bench):
        """Modifying the returned DataFrame doesn't mutate internal state."""
        r1 = fitted_bench.results
        r2 = fitted_bench.results
        r1.loc[0, "total_sec"] = 9999.0
        assert fitted_bench.results.loc[0, "total_sec"] != 9999.0


# ── Persistence ────────────────────────────────────────────────────────────────

def test_save_and_load_params(tmp_path, small_df):
    bench = ScalingBenchmark(small_df, columns=["col_a"], seed=0)
    bench.run(n_vals=[500, 1_000], m_vals=[1], k_vals=[3], verbose=False)
    bench.fit()

    path = tmp_path / "params.json"
    bench.save_params(path)
    assert path.exists()

    loaded = ScalingBenchmark.load_params(path)
    assert set(loaded.keys()) >= {"alpha", "beta", "delta", "gamma"}
    assert loaded["alpha"] == pytest.approx(bench.formula_params["alpha"])


def test_save_before_fit_raises(small_df, tmp_path):
    bench = ScalingBenchmark(small_df, columns=["col_a"])
    with pytest.raises(RuntimeError, match="Call .fit\\(\\)"):
        bench.save_params(tmp_path / "out.json")
