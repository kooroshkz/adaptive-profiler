"""adaptive_profiler.detection — AutoML anomaly detection models and training."""

from .models import SUPPORTED_MODELS
from .trainer import TrainingResult, train_column

__all__ = [
    "SUPPORTED_MODELS",
    "TrainingResult",
    "train_column",
]
