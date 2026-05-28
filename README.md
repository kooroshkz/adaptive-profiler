# adaptive_profiler

AutoML anomaly detection and schema-driven data quality checks for ETL pipelines.

Detects semantic anomalies that rule-based checks miss — stuck sensors, silent feeds, values that are numerically valid but statistically unusual. Each column gets its own model, trained on recent data, configured through a single YAML file.

## Features

- **Two-layer quality checking** — rule-based data contract checks + ML-based anomaly detection, both declared in one YAML file
- **AutoML via Optuna** — searches over IForest, LOF, HBOS, COPOD, and ECOD; selects the best model per column automatically
- **Per-source isolation** — each (partition × column) pair trains its own model; `amsterdam` and `london` never share a model
- **Configurable training window** — train on the N most-recent rows rather than the full history to keep retraining fast
- **Manual model override** — pin a specific algorithm and hyperparameters per column to skip Optuna entirely
- **Tunable flag threshold** — override the binary flagging threshold per column on a validation split without retraining
- **Built-in cost projection** — `ScalingBenchmark` fits T(n,m,k) = α·n^β·m^δ·k^γ so you can predict overhead before committing to production
- **Pipeline-safe output** — failures reported in the output DataFrame, never raised as exceptions

## Install

```bash
pip install adaptive-profiler
pip install "adaptive-profiler[s3]"   # include boto3 for S3 storage
```

For development (from source):

```bash
git clone https://github.com/kooroshkz/adaptive-profiler
cd adaptive-profiler
pip install -e ".[dev]"
```

## Quick start

```python
from adaptive_profiler import Profiler

profiler = Profiler.from_yaml("profiling_schema.yml")

# Train models — one per (city, column) pair
results = profiler.train(partition_key="amsterdam", df=historical_df)
for r in results:
    print(r)

# Score incoming data — returns long-format DataFrame
predictions = profiler.score(partition_key="amsterdam", df=new_df)
print(predictions[predictions["automl_flag"] == 1])

# Rule-based checks only
violations = profiler.check_quality(df=new_df)
```

## Cost projection

```python
from adaptive_profiler import ScalingBenchmark

bench = ScalingBenchmark(df, columns=["temperature_2m", "pressure"])
bench.run(quick=True)   # ~1–2 min
bench.fit()
print(bench.report(target_n=100_000, m=6, k=25))
t = bench.predict(n=100_000, m=6, k=25)
```

## Running tests

```bash
pytest
```

## Module layout

| File | Purpose |
|---|---|
| `profiler.py` | Main `Profiler` class — `train()`, `score()`, `check_quality()` |
| `schema.py` | YAML config dataclasses — `ProfilerConfig`, `ColumnConfig`, `TrainingConfig` |
| `trainer.py` | Optuna HPO loop, manual override path, `TrainingResult` |
| `models.py` | PyOD model registry and Optuna search-space definitions |
| `quality.py` | Rule-based quality checks, `check_dataframe()`, `quality_summary()` |
| `store.py` | `S3Store`, `LocalStore`, `ArtifactStore` protocol |
| `projection.py` | `ScalingBenchmark` — benchmark, fit power-law, predict and report |
