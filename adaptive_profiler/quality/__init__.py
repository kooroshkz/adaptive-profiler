"""adaptive_profiler.quality — Rule-based data contract checks."""

from .checks import QualityViolation, check_dataframe, quality_summary

__all__ = [
    "QualityViolation",
    "check_dataframe",
    "quality_summary",
]
