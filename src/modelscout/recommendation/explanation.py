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
    """Generates human-readable, friendly bullet points explaining why the candidate is recommended."""
    reasons: List[str] = []

    # Memory / Fit reason
    if fit.fit_type == FitType.UNIFIED_MEMORY:
        reasons.append(f"Fits comfortably in {mem.total_required_gb:.1f} GB unified memory")
    elif fit.fit_type == FitType.FULL_GPU:
        reasons.append(f"Fits 100% inside GPU VRAM ({mem.total_required_gb:.1f} GB) for maximum speed")
    elif fit.fit_type == FitType.PARTIAL_OFFLOAD:
        reasons.append(f"Runs smoothly with {int(fit.gpu_layers_pct * 100)}% offloaded to GPU and rest in RAM")

    # Benchmark reason
    if benchmarks.composite_score >= 85.0:
        reasons.append(f"Top-tier intelligence ({benchmarks.composite_score:.1f} on {benchmarks.primary_benchmark})")
    elif benchmarks.composite_score >= 75.0:
        reasons.append(f"Solid capability score ({benchmarks.composite_score:.1f} across benchmark suites)")

    # Speed reason
    if speed.estimated_tok_per_sec >= 20.0:
        reasons.append(f"Super snappy interactive speed ({speed.speed_display})")
    elif speed.estimated_tok_per_sec >= 8.0:
        reasons.append(f"Smooth reading speed ({speed.speed_display})")

    # Evidence reason
    if benchmarks.evidence_type == "direct":
        reasons.append("High-confidence benchmark evidence verified directly on this model")
    elif benchmarks.evidence_type == "variant":
        reasons.append("Verified instruction-tuned performance on official lineage")

    # Capabilities
    if model.capabilities.get("reasoning"):
        reasons.append("Strong multi-step reasoning and mathematical logic")
    if model.capabilities.get("coding"):
        reasons.append("Specialized for coding, debugging, and software workflows")
    if model.capabilities.get("vision"):
        reasons.append("Full vision and image comprehension support")

    # Runtime
    if model.ollama_name:
        reasons.append(f"Ready to run in Ollama: `ollama run {model.ollama_name}`")

    # Artifact
    reasons.append(f"Recommended quant: {quant} (~{mem.weights_gb:.1f} GB download)")

    return reasons


def explain_why_not_recommended(
    model: ModelMetadata,
    fit: FitEvaluation,
    mem: MemoryRequirement,
    speed: SpeedEstimate,
    hardware: SystemHardware,
    missing_requirement: str = "",
) -> str:
    """Generates clear, friendly explanations for excluded models."""
    if missing_requirement:
        return f"Doesn't fit your selected profile requirement: {missing_requirement}."

    if not fit.can_run or fit.fit_type == FitType.UNUSABLE:
        avail = hardware.memory.total_ram_gb
        return f"Needs ~{mem.total_required_gb:.1f} GB of RAM/VRAM, but your machine only has {avail:.1f} GB available."

    if speed.estimated_tok_per_sec < 2.0:
        return f"Technically runnable, but predicted speed ({speed.speed_display}) is too slow for comfortable interactive chat."

    return "Excluded because other models deliver a better balance of capability, fit, and speed for your setup."
