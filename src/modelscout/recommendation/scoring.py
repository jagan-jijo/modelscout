"""Explainable recommendation scoring engine combining benchmark, capability, fit, and speed."""

from typing import Dict
from pydantic import BaseModel, Field

from modelscout.benchmarks.aggregation import AggregatedBenchmarkResult
from modelscout.estimation.fit import FitEvaluation, FitType
from modelscout.estimation.speed import SpeedEstimate
from modelscout.models.metadata import ModelMetadata
from modelscout.models.quantization import get_quant_quality_multiplier
from modelscout.recommendation.profiles import ProfileWeights


class ScoreBreakdown(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    benchmark_quality: float = Field(ge=0, le=100)
    model_capability: float = Field(ge=0, le=100)
    hardware_fit: float = Field(ge=0, le=100)
    speed_score: float = Field(ge=0, le=100)
    evidence_score: float = Field(ge=0, le=100)
    runtime_score: float = Field(ge=0, le=100)


def calculate_recommendation_score(
    model: ModelMetadata,
    benchmarks: AggregatedBenchmarkResult,
    fit: FitEvaluation,
    speed: SpeedEstimate,
    weights: ProfileWeights,
    quantization: str = "Q4_K_M",
) -> ScoreBreakdown:
    """Calculates an explainable 0-100 recommendation score without mystery."""
    if not fit.can_run or fit.fit_type == FitType.UNUSABLE:
        return ScoreBreakdown(
            overall_score=0.0,
            benchmark_quality=0.0,
            model_capability=0.0,
            hardware_fit=0.0,
            speed_score=0.0,
            evidence_score=0.0,
            runtime_score=0.0,
        )

    # 1. Benchmark Quality (0-100), penalized slightly by low bit-depth quantizations
    quant_mult = get_quant_quality_multiplier(quantization)
    bench_qual = round(benchmarks.composite_score * quant_mult, 1)

    # 2. Model Capability (0-100 based on parameter scale & architecture)
    p_b = model.parameters / 1e9
    if p_b >= 65:
        cap = 98.0
    elif p_b >= 30:
        cap = 92.0
    elif p_b >= 14:
        cap = 86.0
    elif p_b >= 7:
        cap = 78.0
    elif p_b >= 3:
        cap = 68.0
    else:
        cap = 58.0
    if model.capabilities.get("reasoning"):
        cap = min(100.0, cap + 3.0)
    if model.capabilities.get("coding") and weights.requires_coding:
        cap = min(100.0, cap + 5.0)

    # 3. Hardware Fit (0-100)
    if fit.fit_type == FitType.FULL_GPU:
        fit_score = 100.0
    elif fit.fit_type == FitType.UNIFIED_MEMORY:
        fit_score = 96.0 if fit.fits_full_gpu else 80.0
    elif fit.fit_type == FitType.MULTI_GPU:
        fit_score = 94.0
    elif fit.fit_type == FitType.PARTIAL_OFFLOAD:
        # Penalize proportional to offloaded layers
        fit_score = round(35.0 + (fit.gpu_layers_pct * 40.0), 1)
    else:  # CPU_ONLY
        fit_score = 30.0

    # 4. Speed Score (0-100)
    sp = speed.estimated_tok_per_sec
    if sp >= 40.0:
        sp_score = 100.0
    elif sp >= 25.0:
        sp_score = 92.0
    elif sp >= 15.0:
        sp_score = 80.0
    elif sp >= 8.0:
        sp_score = 65.0
    elif sp >= 4.0:
        sp_score = 45.0
    elif sp >= 2.0:
        sp_score = 25.0
    else:
        sp_score = 10.0

    # Check minimum speed profile requirement
    if weights.min_speed_threshold > 0 and sp < weights.min_speed_threshold:
        sp_score = max(5.0, sp_score * 0.4)

    # 5. Evidence Quality Score (0-100)
    ev_type = benchmarks.evidence_type.lower()
    if ev_type == "direct":
        ev_score = 100.0
    elif ev_type == "variant":
        ev_score = 90.0
    elif ev_type == "base_model":
        ev_score = 80.0
    elif ev_type == "line_interpolated":
        ev_score = 65.0
    else:
        ev_score = 45.0

    # 6. Runtime Support (0-100)
    rt_score = 95.0
    if model.ollama_name:
        rt_score = 100.0

    # Weighted Overall Score
    overall = (
        (bench_qual * weights.benchmark_weight)
        + (cap * weights.capability_weight)
        + (fit_score * weights.fit_weight)
        + (sp_score * weights.speed_weight)
        + (ev_score * weights.evidence_weight)
        + (rt_score * weights.runtime_weight)
    )
    overall = round(max(0.0, min(100.0, overall)), 1)

    return ScoreBreakdown(
        overall_score=overall,
        benchmark_quality=bench_qual,
        model_capability=round(cap, 1),
        hardware_fit=fit_score,
        speed_score=sp_score,
        evidence_score=ev_score,
        runtime_score=rt_score,
    )
