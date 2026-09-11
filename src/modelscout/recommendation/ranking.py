"""Recommendation ranking pipeline, categorization, and report generation."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from modelscout.benchmarks.engine import BenchmarkEngine
from modelscout.database.repository import DatabaseRepository
from modelscout.estimation.confidence import ConfidenceBreakdown
from modelscout.estimation.fit import FitEvaluation, FitType, evaluate_hardware_fit
from modelscout.estimation.memory import MemoryRequirement, estimate_model_memory
from modelscout.estimation.speed import SpeedEstimate, estimate_generation_speed
from modelscout.hardware.types import SystemHardware
from modelscout.models.discovery import discover_available_models
from modelscout.models.metadata import ModelMetadata
from modelscout.recommendation.explanation import explain_why_not_recommended, explain_why_recommended
from modelscout.recommendation.profiles import RecommendationProfile, get_profile_weights
from modelscout.recommendation.scoring import ScoreBreakdown, calculate_recommendation_score


class ModelRecommendation(BaseModel):
    model_id: str
    display_name: str
    family: str
    parameters: int
    active_parameters: int
    architecture: str
    context_length: int
    quantization: str
    artifact: str
    file_size_bytes: int
    file_size_gb: float

    fit_type: str
    fit_label: str
    vram_required_gb: float
    ram_required_gb: float

    estimated_tok_per_sec: float
    speed_range_tok_per_sec: List[int]
    speed_display: str
    speed_confidence: str
    speed_notes: str

    benchmark_score: float
    benchmark_source: str
    benchmark_confidence: str
    benchmark_evidence: str

    recommendation_score: float
    score_breakdown: ScoreBreakdown

    ollama_name: Optional[str] = None
    nvidia_build_name: Optional[str] = None
    huggingface_id: Optional[str] = None
    huggingface_url: Optional[str] = None
    ollama_url: Optional[str] = None
    nvidia_build_url: Optional[str] = None

    capabilities: Dict[str, bool] = Field(default_factory=dict)
    why_recommended: List[str] = Field(default_factory=list)
    confidence: ConfidenceBreakdown


class ExcludedCandidate(BaseModel):
    model_id: str
    display_name: str
    parameters: int
    required_memory_gb: float
    reason: str


class RecommendationReport(BaseModel):
    hardware: SystemHardware
    hardware_score: int
    profile: str
    recommendations: List[ModelRecommendation] = Field(default_factory=list)
    best_overall: Optional[ModelRecommendation] = None
    best_quality: Optional[ModelRecommendation] = None
    fastest: Optional[ModelRecommendation] = None
    best_coding: Optional[ModelRecommendation] = None
    best_reasoning: Optional[ModelRecommendation] = None
    best_vision: Optional[ModelRecommendation] = None
    excluded: List[ExcludedCandidate] = Field(default_factory=list)
    benchmark_snapshot: str = "2026-09"
    data_snapshot: str = "2026-09-11"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))


def generate_recommendations(
    hardware: SystemHardware,
    profile: str = "general",
    quantization: str = "Q4_K_M",
    top_n: int = 15,
    repo: Optional[DatabaseRepository] = None,
    fit_filter: Optional[str] = None,
    speed_filter: Optional[str] = None,
    min_speed: Optional[float] = None,
    evidence_filter: Optional[str] = None,
    context_length: Optional[int] = None,
    vram_headroom_gb: float = 0.0,
    ram_budget_gb: Optional[float] = None,
) -> RecommendationReport:
    """Executes full decision pipeline: evaluation, filtering, scoring, and ranking."""
    if repo is None:
        repo = DatabaseRepository()

    benchmark_engine = BenchmarkEngine(repo)
    weights = get_profile_weights(profile)
    models = discover_available_models(repo)

    recs: List[ModelRecommendation] = []
    excluded: List[ExcludedCandidate] = []

    for m in models:
        # Profile specific hard filtering
        if weights.requires_vision and not m.capabilities.get("vision"):
            excluded.append(
                ExcludedCandidate(
                    model_id=m.id,
                    display_name=m.display_name,
                    parameters=m.parameters,
                    required_memory_gb=0.0,
                    reason="Model does not support vision/multimodal input as required by profile.",
                )
            )
            continue

        # 1. Memory requirement
        # Find matching artifact or estimate
        target_art = next((a for a in m.artifacts if a.quantization == quantization), None)
        mem_req = estimate_model_memory(m, quantization=quantization, context_tokens=context_length, artifact=target_art)

        # 2. Fit evaluation
        fit = evaluate_hardware_fit(
            mem_req,
            hardware,
            vram_headroom_gb=vram_headroom_gb,
            ram_budget_gb=ram_budget_gb,
        )

        # 3. Speed estimate
        speed = estimate_generation_speed(m, hardware, fit, quantization=quantization)

        # Fit filters
        if fit_filter:
            ff = fit_filter.lower().replace("-", "_")
            if ff in ["full_gpu", "gpu_only"] and fit.fit_type not in [FitType.FULL_GPU, FitType.UNIFIED_MEMORY, FitType.MULTI_GPU]:
                continue
            elif ff == "partial" and fit.fit_type != FitType.PARTIAL_OFFLOAD:
                continue
            elif ff in ["cpu", "cpu_only"] and fit.fit_type != FitType.CPU_ONLY:
                continue

        # Speed filters
        if speed_filter:
            sf = speed_filter.lower()
            if sf == "fast" and speed.estimated_tok_per_sec < 20.0:
                continue
            elif sf == "usable" and speed.estimated_tok_per_sec < 5.0:
                continue
        if min_speed is not None and speed.estimated_tok_per_sec < min_speed:
            continue

        # Exclude unrunnable or excruciatingly slow models from primary recommendations
        if not fit.can_run or fit.fit_type == FitType.UNUSABLE or speed.estimated_tok_per_sec < 1.0:
            reason = explain_why_not_recommended(m, fit, mem_req, speed, hardware)
            excluded.append(
                ExcludedCandidate(
                    model_id=m.id,
                    display_name=m.display_name,
                    parameters=m.parameters,
                    required_memory_gb=mem_req.total_required_gb,
                    reason=reason,
                )
            )
            continue

        # 4. Benchmark evaluation
        bench = benchmark_engine.evaluate_model(m)

        if evidence_filter:
            ef = evidence_filter.lower()
            if ef == "direct" and bench.evidence_type != "direct":
                continue
            elif ef == "base" and bench.evidence_type not in ["direct", "variant", "base_model"]:
                continue

        # 5. Recommendation score
        score_breakdown = calculate_recommendation_score(
            model=m,
            benchmarks=bench,
            fit=fit,
            speed=speed,
            weights=weights,
            quantization=quantization,
        )

        # 6. Explanations
        why = explain_why_recommended(m, bench, fit, mem_req, speed, score_breakdown, quantization)

        conf = ConfidenceBreakdown.create(
            benchmark_conf=bench.confidence,
            speed_conf=speed.speed_confidence,
            runtime_conf="HIGH" if m.ollama_name else "MEDIUM",
        )

        rec = ModelRecommendation(
            model_id=m.id,
            display_name=m.display_name,
            family=m.family,
            parameters=m.parameters,
            active_parameters=m.active_parameters,
            architecture=m.architecture,
            context_length=context_length or m.context_length,
            quantization=quantization,
            artifact=f"{m.display_name}-{quantization}.gguf",
            file_size_bytes=mem_req.weights_bytes,
            file_size_gb=mem_req.weights_gb,
            fit_type=fit.fit_type.value,
            fit_label=fit.fit_label,
            vram_required_gb=fit.vram_used_gb,
            ram_required_gb=fit.ram_used_gb,
            estimated_tok_per_sec=speed.estimated_tok_per_sec,
            speed_range_tok_per_sec=speed.speed_range_tok_per_sec,
            speed_display=speed.speed_display,
            speed_confidence=speed.speed_confidence,
            speed_notes=speed.speed_notes,
            benchmark_score=bench.composite_score,
            benchmark_source=bench.primary_source,
            benchmark_confidence=bench.confidence,
            benchmark_evidence=bench.evidence_type,
            recommendation_score=score_breakdown.overall_score,
            score_breakdown=score_breakdown,
            ollama_name=m.ollama_name,
            nvidia_build_name=m.nvidia_build_name,
            huggingface_id=m.huggingface_id,
            huggingface_url=f"https://huggingface.co/{m.huggingface_id}" if m.huggingface_id else None,
            ollama_url=f"https://ollama.com/library/{m.ollama_name.split(':')[0]}" if m.ollama_name else None,
            nvidia_build_url=f"https://build.nvidia.com/{m.nvidia_build_name}" if m.nvidia_build_name else "https://build.nvidia.com/explore/discover",
            capabilities=m.capabilities,
            why_recommended=why,
            confidence=conf,
        )
        recs.append(rec)

    # Sort candidates by recommendation_score descending
    recs.sort(key=lambda r: r.recommendation_score, reverse=True)

    # Category highlights
    best_overall = recs[0] if recs else None
    best_quality = max(recs, key=lambda r: r.benchmark_score) if recs else None
    fastest = max(recs, key=lambda r: r.estimated_tok_per_sec) if recs else None

    coding_cands = [r for r in recs if r.capabilities.get("coding")]
    best_coding = coding_cands[0] if coding_cands else None

    reasoning_cands = [r for r in recs if r.capabilities.get("reasoning")]
    best_reasoning = reasoning_cands[0] if reasoning_cands else None

    vision_cands = [r for r in recs if r.capabilities.get("vision")]
    best_vision = vision_cands[0] if vision_cands else None

    hw_score_val = hardware.hardware_score.overall_score if hardware.hardware_score else 70

    return RecommendationReport(
        hardware=hardware,
        hardware_score=hw_score_val,
        profile=profile,
        recommendations=recs[:top_n],
        best_overall=best_overall,
        best_quality=best_quality,
        fastest=fastest,
        best_coding=best_coding,
        best_reasoning=best_reasoning,
        best_vision=best_vision,
        excluded=excluded,
    )
