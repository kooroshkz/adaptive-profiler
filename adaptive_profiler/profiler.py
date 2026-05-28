"""Profiler — the main entry point for adaptive_profiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ColumnConfig, ProfilerConfig
from .detection import TrainingResult, train_column
from .quality import check_dataframe, quality_summary
from .storage import ArtifactStore, make_store


class Profiler:
    """Adaptive data profiler: AutoML anomaly detection + rule-based quality checks.

    Typical usage
    -------------
    >>> from adaptive_profiler import Profiler
    >>>
    >>> profiler = Profiler.from_yaml("profiling_schema.yml")
    >>>
    >>> # Train models for one partition (e.g. one city, sensor, or region)
    >>> results = profiler.train(partition_key="amsterdam", df=historical_df)
    >>> for r in results:
    ...     print(r)
    >>>
    >>> # Score new data
    >>> predictions = profiler.score(partition_key="amsterdam", df=new_df)
    >>> print(predictions)
    >>>
    >>> # Run quality checks only (no ML required)
    >>> violations = profiler.check_quality(df=new_df)
    """

    def __init__(self, config: ProfilerConfig, store: ArtifactStore) -> None:
        self._config = config
        self._store = store

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Profiler":
        """Create a Profiler from a profiling_schema.yml file.

        Parameters
        ----------
        path: Path to the user's ``profiling_schema.yml``.
        """
        config = ProfilerConfig.from_yaml(path)
        store = make_store(config.model_store)
        return cls(config, store)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        partition_key: str,
        df: pd.DataFrame,
    ) -> list[TrainingResult]:
        """Train AutoML anomaly detectors for all ``automl: true`` columns.

        One model is trained per column and persisted to the configured store.
        Columns that fail the ``min_train_rows`` threshold are skipped and
        reported in the returned results list rather than raising an exception.

        Parameters
        ----------
        partition_key:
            Identifier for this slice of data — used to namespace the stored
            model (e.g. city name, sensor id, region code).
        df:
            Historical DataFrame.  Include a ``y_true`` column (0/1 labels)
            to enable supervised F2 as the Optuna objective; omit it for a
            fully unsupervised proxy objective.

        Returns
        -------
        List of :class:`~adaptive_profiler.trainer.TrainingResult`, one per
        ``automl: true`` column.
        """
        # ── Apply configurable training window ────────────────────────────────
        window = self._config.training.window_size
        if window is not None and len(df) > window:
            df = df.iloc[-window:].reset_index(drop=True)

        results: list[TrainingResult] = []
        for col_cfg in self._config.automl_columns:
            result, artifact, metadata = train_column(
                df=df,
                col_cfg=col_cfg,
                training_cfg=self._config.training,
                partition_key=partition_key,
            )
            if not result.skipped and artifact is not None:
                self._store.save(
                    partition_key=partition_key,
                    column=col_cfg.name,
                    artifact=artifact,
                    metadata=metadata,
                )
            results.append(result)
        return results

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        partition_key: str,
        df: pd.DataFrame,
        time_column: str = "time",
    ) -> pd.DataFrame:
        """Score new data: AutoML inference + rule-based quality checks.

        For each ``automl: true`` column:
        - Loads the trained model from the store.  If no model has been
          trained yet, ``automl_flag`` and ``automl_score`` are ``None``
          and ``model_available`` is ``False`` — no exception is raised.
        - Runs all schema quality rules.

        Parameters
        ----------
        partition_key: Same identifier used during :meth:`train`.
        df:            New DataFrame to score.
        time_column:   Name of the timestamp column (default ``"time"``).

        Returns
        -------
        Long-format DataFrame with one row per (timestamp × column):

        ============================================================
        Column              Description
        ============================================================
        ``time``            Timestamp (from *time_column*, if present)
        ``partition_key``   Value passed by the caller
        ``column``          Column name
        ``value``           Observed value
        ``automl_flag``     1=anomaly, 0=normal, None=no model
        ``automl_score``    Continuous anomaly score (higher → more anomalous)
        ``model_available`` Whether a trained model was found in the store
        ``quality_violation`` Violated rule string or None when clean
        ============================================================
        """
        times = df[time_column] if time_column in df.columns else pd.Series([None] * len(df))
        records: list[dict[str, Any]] = []

        for col_cfg in self._config.automl_columns:
            col = col_cfg.name
            if col not in df.columns:
                continue

            # ── AutoML inference ─────────────────────────────────────────────
            artifact = self._store.load(partition_key, col)
            automl_flags: np.ndarray | None = None
            automl_scores: np.ndarray | None = None

            if artifact is not None:
                try:
                    X = df[[col]].to_numpy(dtype=float)
                    preprocess = artifact["preprocess"]
                    model = artifact["model"]
                    X_proc = preprocess.transform(X)
                    automl_flags = model.predict(X_proc).astype(int)
                    automl_scores = model.decision_function(X_proc).astype(float)
                except Exception as exc:
                    print(f"[WARN] adaptive_profiler: inference failed for {partition_key}/{col}: {exc}")

            # ── Per-column flag threshold override ───────────────────────────
            # When flag_threshold is set in the schema, use the raw anomaly
            # score rather than PyOD's contamination-based binary prediction.
            # This lets engineers tune sensitivity on a validation split without
            # retraining the model.
            if (
                col_cfg.flag_threshold is not None
                and automl_scores is not None
            ):
                automl_flags = (automl_scores > col_cfg.flag_threshold).astype(int)

            # ── Rule-based checks ─────────────────────────────────────────────
            quality_issues = [
                "; ".join(col_cfg.checks.violations(v)) or None
                for v in df[col]
            ]

            for i in range(len(df)):
                records.append({
                    "time": times.iloc[i],
                    "partition_key": partition_key,
                    "column": col,
                    "value": df[col].iloc[i],
                    "automl_flag": int(automl_flags[i]) if automl_flags is not None else None,
                    "automl_score": float(automl_scores[i]) if automl_scores is not None else None,
                    "model_available": artifact is not None,
                    "quality_violation": quality_issues[i],
                })

        return pd.DataFrame(records)

    # ── Quality-only ──────────────────────────────────────────────────────────

    def check_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run rule-based quality checks without any ML model.

        Useful for fast pre-ingestion validation or when models have not yet
        been trained.

        Returns
        -------
        Violations DataFrame from :func:`~adaptive_profiler.quality.check_dataframe`:
        columns ``column``, ``row_index``, ``rule``, ``value``.
        Empty when the data is clean.
        """
        return check_dataframe(df, list(self._config.columns))

    def quality_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate quality violation counts per column and rule.

        Returns
        -------
        DataFrame with columns ``column``, ``rule``, ``count``, sorted descending.
        """
        return quality_summary(self.check_quality(df))

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def columns(self) -> list[ColumnConfig]:
        """All column configurations from the schema."""
        return list(self._config.columns)

    @property
    def automl_columns(self) -> list[ColumnConfig]:
        """Only the columns with ``automl: true``."""
        return self._config.automl_columns

    def model_status(self, partition_key: str) -> dict[str, bool]:
        """Return a dict of ``{column: model_exists}`` for this partition."""
        return {
            col.name: self._store.exists(partition_key, col.name)
            for col in self._config.automl_columns
        }

    def __repr__(self) -> str:
        return (
            f"Profiler(store={self._store!r}, "
            f"automl_columns={self._config.automl_column_names})"
        )
