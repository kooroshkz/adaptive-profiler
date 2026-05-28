# adaptive_profiler

[![PyPI version](https://img.shields.io/pypi/v/adaptive-profiler.svg)](https://pypi.org/project/adaptive-profiler/)
[![CI](https://github.com/kooroshkz/adaptive-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/kooroshkz/adaptive-profiler/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-profiler.svg)](https://pypi.org/project/adaptive-profiler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AutoML anomaly detection and schema-driven data quality checks for ETL pipelines.

Each column gets its own model, trained on your data, configured through a single YAML file. Catches anomalies that rule-based checks miss — values that are numerically valid but statistically unusual.

## Install

```bash
pip install adaptive-profiler
pip install "adaptive-profiler[s3]"   # include boto3 for S3 storage
```

## Quick start

Define columns in a YAML file:

```yaml
# profiling_schema.yml
version: 1
model_store:
  backend: local
  local_dir: ./models
columns:
  - name: revenue
    automl: true
    checks:
      not_null: true
      min: 0
  - name: user_count
    automl: true
    checks:
      not_null: true
```

Then train and score:

```python
from adaptive_profiler import Profiler

profiler = Profiler.from_yaml("profiling_schema.yml")

# Train — one model per (partition × column) pair
profiler.train(partition_key="region_a", df=historical_df)

# Score new data — returns a DataFrame with anomaly flags
predictions = profiler.score(partition_key="region_a", df=new_df)
print(predictions[predictions["automl_flag"] == 1])

# Rule-based checks only (no ML)
violations = profiler.check_quality(df=new_df)
```

## Features

- **Two-layer checking** — rule-based data contracts + ML anomaly detection, both in one YAML file
- **AutoML via Optuna** — automatically selects the best model (IForest, LOF, HBOS, COPOD, ECOD) per column
- **Per-partition isolation** — each `(partition × column)` pair trains its own model independently
- **Pipeline-safe** — anomalies are returned in the output DataFrame, never raised as exceptions
- **Cost projection** — estimate training overhead before committing to production scale
- **S3 support** — store and load models from S3 with `pip install "adaptive-profiler[s3]"`

## Cost projection

```python
from adaptive_profiler import ScalingBenchmark

bench = ScalingBenchmark(df, columns=["revenue", "user_count"])
bench.run(quick=True)
bench.fit()
print(bench.report(target_n=1_000_000, m=10, k=50))
```

## Development

```bash
git clone https://github.com/kooroshkz/adaptive-profiler
cd adaptive-profiler
pip install -e ".[dev]"
pytest
```
