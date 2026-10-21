# adaptive-profiler

[![PyPI version](https://img.shields.io/pypi/v/adaptive-profiler.svg)](https://pypi.org/project/adaptive-profiler/)
[![CI](https://github.com/kooroshkz/adaptive-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/kooroshkz/adaptive-profiler/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-profiler.svg)](https://pypi.org/project/adaptive-profiler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kooroshkz/adaptive-profiler/blob/main/LICENSE)
[![Paper: BNAIC 2026](https://img.shields.io/badge/paper-BNAIC%202026-2563EB.svg)](https://kooroshkz.github.io/adaptive-data-profiling-etl/)

**Learned, per-column data quality checks for ETL pipelines.**

Rule checks catch values that break a constraint. They do not catch values that are valid but unusual, such as a 0 °C reading that passes a `[-10, 60]` range check in a week when every recent reading was above 25 °C. `adaptive-profiler` adds a second check next to your rules: for each column, an automated search picks and tunes an anomaly detector on that column's own history, and flags values that fall outside what is normal for it. Both layers are declared in one YAML file.

This library is the reference implementation of the BNAIC 2026 paper *Adaptive Data Profiling with ML-Based Quality Checks for ETL Pipeline Quality and Cost Optimization* ([cite it](#citation)).

[![Watch the 5 minute overview](https://raw.githubusercontent.com/kooroshkz/adaptive-data-profiling-etl/main/site/assets/poster.jpg)](https://kooroshkz.com/files/content/AdaptiveProfiler/bnaic2026_video.mp4)

[Watch the overview video (4:50)](https://kooroshkz.com/files/content/AdaptiveProfiler/bnaic2026_video.mp4) · [Project page](https://kooroshkz.github.io/adaptive-data-profiling-etl/) · [Full ETL pipeline and experiments](https://github.com/kooroshkz/adaptive-data-profiling-etl)

## Install

```bash
pip install adaptive-profiler
pip install "adaptive-profiler[s3]"   # adds boto3 for the S3 model store
```

Requires Python 3.10 or newer.

## Quick start

**1. Describe your columns** in `profiling_schema.yml`:

```yaml
version: 1

model_store:
  backend: local
  local_dir: ./models

training:
  n_trials: 25        # search budget per column
  window_size: 5000   # train on the most recent 5000 rows

columns:
  - name: temperature_2m
    automl: true      # learned check on
    checks:
      not_null: true
      range: [-10, 60]

  - name: precipitation
    automl: false     # rule checks only
    checks:
      not_null: true
      range: [0, 300]
```

**2. Train and score:**

```python
from adaptive_profiler import Profiler

profiler = Profiler.from_yaml("profiling_schema.yml")

# One model per (partition, column). A partition is any source: a city, sensor, region.
results = profiler.train(partition_key="amsterdam", df=history_df)
for r in results:
    print(r)
# [OK]   amsterdam/temperature_2m: model=LOF ...
# [SKIP] amsterdam/...: only 42 non-null rows (min=500)

# Learned check and rule checks on new data
scores = profiler.score(partition_key="amsterdam", df=new_df)
print(scores[scores["automl_flag"] == 1])

# Rule checks only, no models needed
violations = profiler.check_quality(new_df)
summary = profiler.quality_summary(new_df)
```

`score` never raises on bad data. It returns a long-format DataFrame with one row per (timestamp, column):

| Column | Meaning |
|:-|:-|
| `time`, `partition_key`, `column`, `value` | where the value came from |
| `automl_flag` | `1` anomaly, `0` normal, `None` when no model is trained |
| `automl_score` | continuous anomaly score (higher is more unusual) |
| `model_available` | whether a trained model was found |
| `quality_violation` | the violated rule, or `None` |

## How it works

```
source ──▶ [ rule checks ──▶ learned check ] ──▶ transform ──▶ load
                       extract stage
```

- **Two layers, one schema.** Rule checks (`type`, `not_null`, `range`) and the learned check live side by side, so the profiler fits where a data contract already exists.
- **One detector per column and source.** Each `(partition_key, column)` pair gets its own model, trained on its own recent history.
- **Automated selection.** An [Optuna](https://optuna.org) TPE search chooses among five [PyOD](https://pyod.readthedocs.io) detectors (`IForest`, `LOF`, `ECOD`, `HBOS`, `COPOD`) and their hyperparameters.
- **No labels required.** Without labels, the search uses a label-free proxy objective (the spread of anomaly scores on a validation split). If your data has a `y_true` column (0/1), F2 on a validation split is used instead.
- **Run it at ingestion.** Checking before data is transformed means a bad value can still be isolated, before it is folded into an aggregate.

## Configuration

Per-column options, all optional except `name`:

```yaml
columns:
  - name: surface_pressure
    description: "Surface pressure (hPa)"
    automl: true
    flag_threshold: 0.45    # flag on raw score; tune without retraining
    checks:
      type: float
      not_null: true
      range: [870, 1085]

  - name: soil_moisture
    automl: true
    model: LOF              # pin a detector and skip the search
    hyperparameters:
      n_neighbors: 20
      contamination: 0.02
```

Training options: `n_trials`, `window_size`, `min_train_rows` (columns with fewer valid rows are skipped and reported) and `seed`. For S3, set `backend: s3`, `bucket` and `prefix`; credentials come from the standard boto3 chain. See the [User Guide](https://github.com/kooroshkz/adaptive-profiler/blob/main/USERGUIDE.md) for the full reference.

## Estimate the cost before you deploy

Training time follows an empirical power law in rows `n`, profiled columns `m` and search trials `k`:

```
T(n, m, k) = α · n^β · m^δ · k^γ
```

Fit it on a sample, then project to production scale:

```python
from adaptive_profiler import ScalingBenchmark

bench = ScalingBenchmark(sample_df, columns=["temperature_2m", "surface_pressure"])
bench.run(quick=True)          # small benchmark grid, about 1 to 2 minutes
bench.fit()

seconds = bench.predict(n=100_000, m=6, k=25)
print(bench.report(target_n=100_000, m=6, k=25))
bench.save_params("formula_params.json")
```

## When to use it

From the evaluation in the paper (hourly weather in five cities and British electricity demand, with injected anomalies as ground truth):

| | Result |
|:-|:-|
| Works well | Columns with tight, predictable distributions. Surface pressure and soil moisture reach F2 above 0.9. |
| Works poorly | Skewed or noisy columns. Precipitation stays below F2 0.21; leave it as `automl: false`. |
| Versus rules | Amsterdam temperature: a `[-10, 60]` rule admits all 74 injected anomalies; the learned check catches 53. |
| Cost | The cost model explains about 86% of the variance over about 90 runs. Calibrated on a quarter of the data, the projection lands within 8.9% of the measured time. |
| Versus neural | Under the same search and budget, these classical detectors beat an autoencoder search in 29 of 30 columns: 4.1 times the mean F2 at about one ninth of the time. |

A practical routine: **pilot** on a sample, **project** the cost with `ScalingBenchmark`, **enable** `automl: true` only on columns where it helps, and **retrain** on a schedule as the source changes.

Known limits: each column is modelled on its own, so anomalies that only show up across several columns together are not detected; and anomalies already present in the training history are learned as normal.

## Development

```bash
git clone https://github.com/kooroshkz/adaptive-profiler
cd adaptive-profiler
pip install -e ".[dev]"
pytest
```

The full ETL pipeline (Airflow, dbt, DuckDB, S3) and the experiment scripts from the paper are in [adaptive-data-profiling-etl](https://github.com/kooroshkz/adaptive-data-profiling-etl).

## Citation

If you use this library or build on the method, please cite:

```bibtex
@inproceedings{komeilizadeh2026adaptive,
  title     = {Adaptive Data Profiling with ML-Based Quality Checks
               for ETL Pipeline Quality and Cost Optimization},
  author    = {Komeili Zadeh, Koorosh and Visser, Joost and van Rijn, Jan N.},
  booktitle = {Benelux Conference on Artificial Intelligence (BNAIC)},
  year      = {2026}
}
```

Koorosh Komeili Zadeh, Joost Visser, Jan N. van Rijn. Leiden Institute of Advanced Computer Science (LIACS), Leiden University.

## License

[MIT](https://github.com/kooroshkz/adaptive-profiler/blob/main/LICENSE)
