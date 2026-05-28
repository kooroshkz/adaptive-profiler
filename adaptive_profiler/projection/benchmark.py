"""
adaptive_profiler.projection
════════════════════════════
Scaling benchmark and cost-projection tool.

Benchmarks the AutoML training loop across a grid of (n rows, m columns,
k Optuna trials) and fits the empirical power-law formula::

    T(n, m, k) = α · n^β · m^δ · k^γ

Engineers can run this on a data sample *before* committing to production so
they know whether the computational overhead is acceptable at their full scale.

Quick start
-----------
>>> from adaptive_profiler import ScalingBenchmark
>>>
>>> bench = ScalingBenchmark(df, columns=["temperature_2m", "pressure"])
>>> bench.run(quick=True)          # ~1–2 min on a laptop (~15 cells)
>>> bench.fit()
>>> print(bench.report(target_n=100_000, m=6, k=25))
>>> t = bench.predict(n=100_000, m=6, k=25)
>>> print(f"Estimated training time: {t:.1f} s")

Loading from parquet files directly
------------------------------------
>>> bench = ScalingBenchmark.from_parquet(
...     "airflow/data/raw/city=amsterdam/hourly_*.parquet",
...     columns=["temperature_2m", "precipitation"],
... )
>>> bench.run()
>>> bench.fit()
>>> print(bench.report())
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _default_params(model_name: str) -> dict[str, Any]:
    """Representative mid-range hyperparameters for timing benchmarks."""
    if model_name == "IForest":
        return {"contamination": 0.05, "n_estimators": 100, "max_samples": 0.8}
    if model_name == "LOF":
        return {"contamination": 0.05, "n_neighbors": 30, "leaf_size": 30}
    if model_name == "HBOS":
        return {"contamination": 0.05, "n_bins": 20, "alpha": 0.1, "tol": 0.5}
    if model_name == "COPOD":
        return {"contamination": 0.05}
    if model_name == "ECOD":
        return {"contamination": 0.05}
    raise ValueError(f"Unknown model: {model_name!r}")


def _time_one_trial(X: np.ndarray, model_name: str) -> float:
    """Fit one detector on *X*, return wall-clock seconds."""
    from ..detection.models import _build_model
    model = _build_model(model_name, _default_params(model_name))
    t0 = time.perf_counter()
    model.fit(X)
    return time.perf_counter() - t0


def _run_cell(
    X_full: np.ndarray,
    n: int,
    m: int,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """Run *k* timing trials on an n×m subsample of *X_full*.

    Models are cycled round-robin across SUPPORTED_MODELS — identical to the
    production AutoML Optuna search distribution.
    """
    from ..detection.models import SUPPORTED_MODELS

    rng = np.random.default_rng(seed)
    row_idx = rng.choice(len(X_full), size=min(n, len(X_full)), replace=False)
    col_idx = rng.choice(X_full.shape[1], size=min(m, X_full.shape[1]), replace=False)
    X = X_full[np.ix_(row_idx, col_idx)]

    model_cycle = list(SUPPORTED_MODELS)
    trial_times: list[float] = []

    def _objective(trial: optuna.Trial) -> float:
        mname = model_cycle[trial.number % len(model_cycle)]
        t = _time_one_trial(X, mname)
        trial_times.append(t)
        return float(rng.random())  # dummy objective — we only need wall time

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )
    study.optimize(_objective, n_trials=k, show_progress_bar=False)

    return {
        "n": n,
        "m": m,
        "k": k,
        "total_sec": float(sum(trial_times)),
        "mean_trial_sec": float(np.mean(trial_times)),
    }


def _fit_log_linear(df: pd.DataFrame) -> dict[str, Any]:
    """Fit T(n,m,k) = α·n^β·m^δ·k^γ via OLS on log-transformed observations.

    Returns a dict with keys: alpha, beta, delta, gamma, r2, n_obs,
    beta_ci95, delta_ci95, gamma_ci95 (each ci95 is a [lo, hi] list).
    """
    log_T = np.log(df["total_sec"].values.astype(float))
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["n"].values.astype(float)),
        np.log(df["m"].values.astype(float)),
        np.log(df["k"].values.astype(float)),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_T, rcond=None)
    log_alpha, beta, delta, gamma = coeffs
    alpha = float(np.exp(log_alpha))

    T_pred = X @ coeffs
    ss_res = float(np.sum((log_T - T_pred) ** 2))
    ss_tot = float(np.sum((log_T - log_T.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 95% bootstrap confidence intervals (1 000 resamples)
    rng = np.random.default_rng(0)
    boot: dict[str, list[float]] = {"beta": [], "delta": [], "gamma": []}
    for _ in range(1000):
        idx = rng.integers(0, len(df), len(df))
        try:
            cb = np.linalg.lstsq(X[idx], log_T[idx], rcond=None)[0]
            boot["beta"].append(float(cb[1]))
            boot["delta"].append(float(cb[2]))
            boot["gamma"].append(float(cb[3]))
        except Exception:
            pass

    def _ci95(samples: list[float]) -> list[float]:
        return [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ]

    return {
        "alpha": alpha,
        "beta": float(beta),
        "delta": float(delta),
        "gamma": float(gamma),
        "r2": float(r2),
        "n_obs": int(len(df)),
        "beta_ci95": _ci95(boot["beta"]),
        "delta_ci95": _ci95(boot["delta"]),
        "gamma_ci95": _ci95(boot["gamma"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class ScalingBenchmark:
    """Benchmark the AutoML training loop and project cost at any data scale.

    Parameters
    ----------
    df:      DataFrame containing at least the columns listed in *columns*.
    columns: Column names to include in the sweep.  All must be numeric.
    seed:    Random seed for reproducibility.

    Examples
    --------
    >>> bench = ScalingBenchmark(df, columns=["temperature_2m", "pressure"])
    >>> bench.run(quick=True)
    >>> bench.fit()
    >>> t = bench.predict(n=100_000, m=6, k=25)
    >>> print(bench.report(target_n=100_000, m=6, k=25))
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns: list[str],
        seed: int = 42,
    ) -> None:
        available = [c for c in columns if c in df.columns]
        if not available:
            raise ValueError(
                f"None of the requested columns {columns!r} were found in the DataFrame."
            )
        missing = [c for c in columns if c not in df.columns]
        if missing:
            import warnings
            warnings.warn(
                f"Columns not found in DataFrame and will be skipped: {missing!r}",
                stacklevel=2,
            )

        # Pre-process once; benchmark cells subsample from this matrix
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_raw = df[available].to_numpy(dtype=float)
        self._X: np.ndarray = scaler.fit_transform(imp.fit_transform(X_raw))
        self._columns: list[str] = available
        self._seed: int = seed
        self._results: pd.DataFrame | None = None
        self._params: dict[str, Any] | None = None

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        path: str,
        columns: list[str] | None = None,
        seed: int = 42,
    ) -> "ScalingBenchmark":
        """Create a ScalingBenchmark by loading data from parquet files.

        Parameters
        ----------
        path:    File path or glob (e.g. ``"data/raw/**/*.parquet"``).
                 Passed directly to DuckDB's ``read_parquet``.
        columns: Column names to benchmark.  When ``None``, up to six numeric
                 columns are auto-detected.
        seed:    Random seed.

        Requires ``duckdb`` (``pip install duckdb``).
        """
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for parquet loading.  "
                "Install it with:  pip install duckdb"
            ) from exc

        con = duckdb.connect(":memory:")
        df = con.execute(
            f"SELECT * FROM read_parquet('{path}', union_by_name=true)"
        ).fetchdf()
        con.close()

        if columns is None:
            numeric = (
                df.select_dtypes(include=[float, int, np.floating, np.integer])
                .columns.tolist()
            )
            columns = numeric[:6]
            if not columns:
                raise ValueError(
                    "No numeric columns found. Specify columns= explicitly."
                )

        return cls(df, columns=columns, seed=seed)

    # ── Benchmark sweep ───────────────────────────────────────────────────────

    def run(
        self,
        quick: bool = False,
        n_vals: list[int] | None = None,
        m_vals: list[int] | None = None,
        k_vals: list[int] | None = None,
        verbose: bool = True,
    ) -> "ScalingBenchmark":
        """Run the benchmark grid and record training times.

        Parameters
        ----------
        quick:   Small fast grid (~15 cells, ~1–2 min) instead of the full
                 grid (~90 cells, ~10–20 min).
        n_vals:  Row counts to sweep (defaults depend on *quick*).
        m_vals:  Column counts.  Values are clipped to the available columns.
        k_vals:  Optuna trial counts.
        verbose: Print each cell's progress to stdout.

        Returns
        -------
        ``self`` — allows method chaining: ``bench.run().fit()``.
        """
        max_m = self._X.shape[1]

        if n_vals is None:
            n_vals = (
                [500, 1_000, 2_500, 5_000, 10_000]
                if quick
                else [500, 1_000, 2_500, 5_000, 10_000, 20_000]
            )
        if m_vals is None:
            raw_m = [1, 3, min(6, max_m)] if quick else [1, 2, 3, 4, min(6, max_m)]
            m_vals = sorted({max(1, min(v, max_m)) for v in raw_m})
        if k_vals is None:
            k_vals = [10, 20] if quick else [10, 20, 30]

        grid = list(product(n_vals, m_vals, k_vals))
        rows: list[dict[str, Any]] = []

        for i, (n, m, k) in enumerate(grid, 1):
            if verbose:
                print(
                    f"  [{i:>3}/{len(grid)}]  n={n:>6,}  m={m}  k={k} … ",
                    end="",
                    flush=True,
                )
            cell = _run_cell(self._X, n=n, m=m, k=k, seed=self._seed)
            rows.append(cell)
            if verbose:
                print(f"{cell['total_sec']:.2f}s")

        self._results = pd.DataFrame(rows)
        return self

    # ── Formula fitting ───────────────────────────────────────────────────────

    def fit(self) -> dict[str, Any]:
        """Fit T(n,m,k) = α·n^β·m^δ·k^γ from the benchmark observations.

        Uses ordinary least squares on log-transformed data.  Also computes
        R² and 95% bootstrap confidence intervals for each exponent.

        Returns
        -------
        Dict with keys ``alpha``, ``beta``, ``delta``, ``gamma``, ``r2``,
        ``n_obs``, ``beta_ci95``, ``delta_ci95``, ``gamma_ci95``.

        Raises
        ------
        RuntimeError: if :meth:`run` has not been called yet.
        """
        if self._results is None:
            raise RuntimeError(
                "No benchmark data.  Call .run() before .fit()."
            )
        self._params = _fit_log_linear(self._results)
        return dict(self._params)

    @property
    def formula_params(self) -> dict[str, Any] | None:
        """The fitted parameters dict, or ``None`` if :meth:`fit` hasn't run."""
        return dict(self._params) if self._params else None

    @property
    def results(self) -> pd.DataFrame | None:
        """Raw benchmark timing DataFrame, or ``None`` before :meth:`run`."""
        return self._results.copy() if self._results is not None else None

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, n: int, m: int, k: int) -> float:
        """Predict training time (seconds) for a given (n, m, k) configuration.

        This is the time for **one partition** (one city, sensor, or region).
        Multiply by the number of partitions for the full pipeline estimate.

        Parameters
        ----------
        n: Training rows per partition.
        m: Number of columns with ``automl: true``.
        k: Optuna trial budget (``n_trials`` in the YAML schema).

        Raises
        ------
        RuntimeError: if :meth:`fit` has not been called yet.
        """
        if self._params is None:
            raise RuntimeError(
                "No fitted parameters.  Call .run() then .fit() before .predict()."
            )
        p = self._params
        return float(
            p["alpha"] * (n ** p["beta"]) * (m ** p["delta"]) * (k ** p["gamma"])
        )

    # ── Report ────────────────────────────────────────────────────────────────

    def report(
        self,
        target_n: int | None = None,
        m: int | None = None,
        k: int = 25,
        scale_factors: list[int] | None = None,
    ) -> str:
        """Return a formatted report: formula, fit quality, and projections.

        Parameters
        ----------
        target_n:       Reference row count (default: max n in benchmark).
        m:              Column count for projections (default: max m benchmarked).
        k:              Trial count for projections (default: 25).
        scale_factors:  Dataset scale multiples to project (default: 1,2,5,10,50,100).

        Raises
        ------
        RuntimeError: if :meth:`fit` has not been called.
        """
        if self._params is None:
            raise RuntimeError("Call .run() then .fit() before .report().")
        if self._results is None:
            raise RuntimeError("No benchmark results.")

        p = self._params
        HR = "─" * 62
        lines: list[str] = [
            "",
            "═" * 62,
            "  Adaptive Profiler — AutoML Scaling Formula",
            "═" * 62,
            "",
            "  T(n, m, k)  ≈  α · n^β · m^δ · k^γ",
            "",
            f"  α  (baseline, hardware-specific)  = {p['alpha']:.4e} s",
            f"  β  (row scaling exponent)          = {p['beta']:.4f}"
            f"   95% CI [{p['beta_ci95'][0]:.3f}, {p['beta_ci95'][1]:.3f}]",
            f"  δ  (column scaling exponent)       = {p['delta']:.4f}"
            f"   95% CI [{p['delta_ci95'][0]:.3f}, {p['delta_ci95'][1]:.3f}]",
            f"  γ  (trial budget exponent)         = {p['gamma']:.4f}"
            f"   95% CI [{p['gamma_ci95'][0]:.3f}, {p['gamma_ci95'][1]:.3f}]",
            "",
            f"  R²  = {p['r2']:.4f}   ({p['n_obs']} benchmark observations)",
            "",
        ]

        beta = p["beta"]
        if beta < 1.0:
            verdict = f"sub-linear (β={beta:.3f} < 1) — scales well ✓"
        elif beta < 1.2:
            verdict = f"near-linear (β={beta:.3f}) — moderate cost growth"
        else:
            verdict = f"super-linear (β={beta:.3f} > 1) — plan retraining budget ⚠"
        lines += [f"  Row scaling: {verdict}", ""]

        # Projection table
        if target_n is None:
            target_n = int(self._results["n"].max())
        if m is None:
            m = int(self._results["m"].max())
        if scale_factors is None:
            scale_factors = [1, 2, 5, 10, 50, 100]

        t_ref = self.predict(target_n, m, k)
        lines += [
            HR,
            f"  Projections  (m={m} columns, k={k} trials)",
            HR,
            f"  {'Scale':<8}  {'Rows':>10}    {'Est. time':>12}    {'vs ref':>8}",
            f"  {'─'*8}  {'─'*10}    {'─'*12}    {'─'*8}",
        ]

        for s in scale_factors:
            n_s = target_n * s
            t_s = self.predict(n_s, m, k)
            if t_s < 60:
                label = f"{t_s:.1f}s"
            elif t_s < 3600:
                label = f"{t_s / 60:.1f} min"
            else:
                label = f"{t_s / 3600:.1f} hr"
            tag = "  ← current scale" if s == 1 else ""
            lines.append(
                f"  {s}×{'':<6}  {n_s:>10,}    {label:>12}    {t_s/t_ref:>6.2f}×{tag}"
            )

        lines += [
            HR,
            f"  2× more data costs {2 ** beta:.2f}× more time  (β={beta:.3f})",
            HR,
            "",
        ]
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_params(self, path: str | Path) -> None:
        """Save fitted formula parameters to a JSON file.

        Parameters
        ----------
        path: Destination file path (e.g. ``"projection/formula_params.json"``).

        Raises
        ------
        RuntimeError: if :meth:`fit` has not been called yet.
        """
        if self._params is None:
            raise RuntimeError("Call .fit() before .save_params().")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._params, f, indent=2)

    @classmethod
    def load_params(cls, path: str | Path) -> dict[str, Any]:
        """Load pre-fitted formula parameters from a JSON file.

        Returns the params dict.  Pass to ``ScalingBenchmark`` via manual
        assignment if you want to call :meth:`predict` without re-running::

            params = ScalingBenchmark.load_params("formula_params.json")
            # predict directly
            alpha, beta, delta, gamma = (
                params["alpha"], params["beta"], params["delta"], params["gamma"]
            )
            t = alpha * (n ** beta) * (m ** delta) * (k ** gamma)
        """
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        fitted = "fitted" if self._params else "not fitted"
        n_cells = len(self._results) if self._results is not None else 0
        return (
            f"ScalingBenchmark("
            f"columns={self._columns!r}, "
            f"n_rows={self._X.shape[0]:,}, "
            f"benchmark_cells={n_cells}, "
            f"formula={fitted})"
        )
