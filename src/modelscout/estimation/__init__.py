"""Estimation package for memory, KV cache, hardware fit, speed, and confidence."""

from modelscout.estimation.confidence import ConfidenceBreakdown
from modelscout.estimation.fit import FitEvaluation, FitType, evaluate_hardware_fit
from modelscout.estimation.kv_cache import calculate_kv_cache_bytes, estimate_kv_cache_from_params
from modelscout.estimation.memory import MemoryRequirement, estimate_model_memory
from modelscout.estimation.speed import SpeedEstimate, estimate_generation_speed

__all__ = [
    "calculate_kv_cache_bytes",
    "estimate_kv_cache_from_params",
    "MemoryRequirement",
    "estimate_model_memory",
    "FitType",
    "FitEvaluation",
    "evaluate_hardware_fit",
    "SpeedEstimate",
    "estimate_generation_speed",
    "ConfidenceBreakdown",
]
