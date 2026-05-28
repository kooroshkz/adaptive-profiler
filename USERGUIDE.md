# User Guide

## YAML configuration reference

A single `profiling_schema.yml` controls both layers for every column.

```yaml
version: 1

model_store:
  backend: local          # "local" or "s3"
  local_dir: ./models     # used when backend=local
  bucket: my-bucket       # required when backend=s3
  prefix: models/v1       # S3 key prefix

training:
  n_trials: 25            # Optuna search budget (per column)
  min_train_rows: 500     # skip column if fewer valid rows exist
  window_size: 5000       # train on the most-recent N rows (omit = use all)
  seed: 42

columns:
  # ── Standard AutoML ───────────────────────────────────────────────────────
  - name: temperature_2m
    description: "Air temperature (°C)"
    automl: true
    checks:
      not_null: true
      range: [-60, 60]

  # ── AutoML with a custom flag threshold ───────────────────────────────────
  # Threshold applies to the raw anomaly score from decision_function().
  # Tune on a validation split; does not require retraining.
  - name: surface_pressure
    automl: true
    flag_threshold: 0.45
    checks:
      not_null: true

  # ── Manual model override (skip Optuna) ───────────────────────────────────
  # Supported models: IForest, LOF, HBOS, COPOD, ECOD
  # contamination defaults to 0.05 if omitted from hyperparameters.
  - name: precipitation
    automl: true
    model: HBOS
    hyperparameters:
      n_bins: 15
      contamination: 0.03
    checks:
      not_null: true
      range: [0, 300]

  # ── Rule checks only ─────────────────────────────────────────────────────
  - name: wind_speed
    automl: false
    checks:
      not_null: true
      range: [0, 200]
```

---

## Training

```python
from adaptive_profiler import Profiler

profiler = Profiler.from_yaml("profiling_schema.yml")
results = profiler.train(partition_key="amsterdam", df=df)

# Each TrainingResult tells you what happened per column:
for r in results:
    print(r)
# [OK]   amsterdam/temperature_2m: model=LOF val_f2=0.923 n_rows=5000
# [SKIP] amsterdam/precipitation: only 42 non-null rows (min=500)
```

**Include `y_true` (0/1) in your DataFrame** to use F2 as the Optuna objective.  
Without labels the profiler falls back to a proxy objective (score variance).

**Training window** — pass a DataFrame covering full history; `window_size` in the YAML slices it to the last N rows automatically.

---

## Scoring

```python
predictions = profiler.score(partition_key="amsterdam", df=new_df)
```

Returns a long-format DataFrame with one row per `(timestamp × column)`:

| Column | Description |
|---|---|
| `time` | Timestamp |
| `partition_key` | Partition identifier |
| `column` | Column name |
| `value` | Observed value |
| `automl_flag` | `1` = anomaly, `0` = normal, `None` = no model |
| `automl_score` | Continuous anomaly score (higher → more anomalous) |
| `model_available` | Whether a trained model was found |
| `quality_violation` | Violated rule string, or `None` when clean |

---

## Rule-based checks only

```python
violations = profiler.check_quality(df)   # violations DataFrame
summary = profiler.quality_summary(df)    # per-column violation counts
```

---

## Cost projection

```python
from adaptive_profiler import ScalingBenchmark

# Run on a representative sample of your data
bench = ScalingBenchmark(df, columns=["temperature_2m", "pressure"])
bench.run(quick=True)    # ~15 cells, ~1–2 min
bench.fit()

# Predict time for your full production scale
t = bench.predict(n=100_000, m=6, k=25)   # seconds, one partition
print(bench.report(target_n=100_000, m=6, k=25))

# Save for later use
bench.save_params("projection/formula_params.json")
```

Formula: `T(n, m, k) = α · n^β · m^δ · k^γ`  
where n = rows, m = automl columns, k = Optuna trials.

---

## Storage backends

**Local** (development / CI):
```yaml
model_store:
  backend: local
  local_dir: ./models
```

**S3** (production):
```yaml
model_store:
  backend: s3
  bucket: my-data-bucket
  prefix: adaptive_profiler/models
```
Credentials are read from the standard boto3 chain (env vars, `~/.aws/credentials`, IAM role).

---

## Running tests

```bash
pip install pytest
pytest adaptive_profiler/tests/ -v
```

Fast: the test suite uses `n_trials=2` and 200-row DataFrames. Full suite runs in under 2 minutes on a laptop.
