"""adaptive_profiler — AutoML anomaly detection and schema-driven data quality profiling.

Quick start
-----------
>>> from adaptive_profiler import Profiler
>>>
>>> profiler = Profiler.from_yaml("profiling_schema.yml")
>>>
>>> # Train per-column models (partition_key can be city, sensor, region, …)
>>> results = profiler.train(partition_key="amsterdam", df=historical_df)
>>>
>>> # Score new data: returns anomaly flags + rule violations
>>> predictions = profiler.score(partition_key="amsterdam", df=new_df)
>>>
>>> # Rule-based quality checks only (no ML)
>>> violations = profiler.check_quality(df=new_df)

Cost projection before production deployment
---------------------------------------------
>>> from adaptive_profiler import ScalingBenchmark
>>>
>>> bench = ScalingBenchmark(df, columns=["temperature_2m", "pressure"])
>>> bench.run(quick=True)   # benchmark a small grid (~1–2 min)
>>> bench.fit()             # fit T(n, m, k) = α · n^β · m^δ · k^γ
>>> print(bench.report(target_n=100_000, m=6, k=25))
>>> t = bench.predict(n=100_000, m=6, k=25)
"""

from .config import ColumnChecks, ColumnConfig, ModelStoreConfig, ProfilerConfig, TrainingConfig
from .detection import SUPPORTED_MODELS, TrainingResult
from .profiler import Profiler
from .projection import ScalingBenchmark
from .quality import QualityViolation, check_dataframe, quality_summary
from .storage import ArtifactStore, LocalStore, S3Store, make_store

__version__ = "0.2.0"
__all__ = [
    # Main entry point
    "Profiler",
    # Cost projection
    "ScalingBenchmark",
    # Config / schema
    "ProfilerConfig",
    "ColumnConfig",
    "ColumnChecks",
    "ModelStoreConfig",
    "TrainingConfig",
    # Storage
    "ArtifactStore",
    "S3Store",
    "LocalStore",
    "make_store",
    # Training
    "TrainingResult",
    # Quality
    "QualityViolation",
    "check_dataframe",
    "quality_summary",
    # Models
    "SUPPORTED_MODELS",
]
