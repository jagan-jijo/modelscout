"""Explanation generator explaining Why and Why Not for recommendations."""

from typing import List
from modelscout.benchmarks.aggregation import AggregatedBenchmarkResult
from modelscout.estimation.fit import FitEvaluation, FitType
from modelscout.estimation.memory import MemoryRequirement
from modelscout.estimation.speed import SpeedEstimate
from modelscout.hardware.types import SystemHardware
from modelscout.models.metadata import ModelMetadata
from modelscout.recommendation.scoring import ScoreBreakdown


def explain_why_recommended(
    model: ModelMetadata,
    benchmarks: AggregatedBenchmarkResult,
    fit: FitEvaluation,
    mem: MemoryRequirement,
    speed: SpeedEstimate,
    score: ScoreBreakdown,
    quant: str,
) -> List[str]:
    """Generates human-readable bullet points explaining why the candidate is recommended."""
    reasons: List[str] = []

    # Memory / Fit reason
    if fit.fit_type == FitType.UNIFIED_MEMORY:
        reasons.append(f"Fits comfortably in {mem.total_required_gb} GB unified memory")
    elif fit.fit_type == FitType.FULL_GPU:
        reasons.append(f"Fits entirely in VRAM ({mem.total_required_gb} GB required)")
    elif fit.fit_type == FitType.PARTIAL_OFFLOAD:
        reasons.append(f"Usable with partial GPU offload ({int(fit.gpu_layers_pct * 100)}% on GPU)")

    # Benchmark reason
    if benchmarks.composite_score >= 85.0:
        reasons.append(f"Top-tier benchmark performance ({benchmarks.composite_score:.1f} on {benchmarks.primary_benchmark})")
    elif benchmarks.composite_score >= 75.0:
        reasons.append(f"Solid benchmark capability score ({benchmarks.composite_score:.1f})")

    # Speed reason
    if speed.estimated_tok_per_sec >= 20.0:
        reasons.append(f"Fast interactive speed ({speed.speed_display})")
    elif speed.estimated_tok_per_sec >= 8.0:
        reasons.append(f"Acceptable generation speed ({speed.speed_display})")

    # Evidence reason
    if benchmarks.evidence_type == "direct":
        reasons.append("High-confidence direct benchmark evidence")
    elif benchmarks.evidence_type == "variant":
        reasons.append("Verified instruction-tuned variant evidence")

    # Capabilities
    if model.capabilities.get("reasoning"):
        reasons.append("Trained with deep reasoning capabilities")
    if model.capabilities.get("coding"):
        reasons.append("Specialized for coding and agentic tasks")
    if model.capabilities.get("vision"):
        reasons.append("Multimodal image understanding support")

    # Runtime
    if model.ollama_name:
        reasons.append(f"Available directly in Ollama (`ollama run {model.ollama_name}`)")

    # Artifact
    reasons.append(f"Recommended artifact: {quant} ({mem.weights_gb} GB weights)")

    return reasons


def explain_why_not_recommended(
    model: ModelMetadata,
    fit: FitEvaluation,
    mem: MemoryRequirement,
    speed: SpeedEstimate,
    hardware: SystemHardware,
    missing_requirement: str = "",
) -> str:
    """Generates clear, helpful explanation for excluded models."""
    if missing_requirement:
        return f"Does not meet profile requirement: {missing_requirement}."

    if not fit.can_run or fit.fit_type == FitType.UNUSABLE:
        avail = hardware.memory.total_ram_gb
        return f"Exceeds memory: Requires ~{mem.total_required_gb:.1f} GB, but system only has {avail:.1f} GB available."

    if speed.estimated_tok_per_sec < 2.0:
        return f"Technically runnable with heavy offload, but predicted speed is too slow ({speed.speed_display}) for comfortable interactive use."

    return "Excluded based on lower overall balance of capability, fit, and generation speed compared to top recommendations."
