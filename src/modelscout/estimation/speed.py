"""Memory bandwidth-based performance and speed range estimation without false precision."""

from typing import List, Optional, Tuple
from pydantic import BaseModel

from modelscout.estimation.fit import FitEvaluation, FitType
from modelscout.hardware.types import SystemHardware
from modelscout.models.metadata import ModelMetadata
from modelscout.models.quantization import get_bits_per_weight


class SpeedEstimate(BaseModel):
    estimated_tok_per_sec: float
    speed_range_tok_per_sec: List[int]  # e.g. [22, 28]
    speed_display: str  # e.g. "~22–28 tok/s"
    speed_confidence: str  # HIGH, MEDIUM, LOW
    speed_notes: str


def estimate_generation_speed(
    model: ModelMetadata,
    hardware: SystemHardware,
    fit: FitEvaluation,
    quantization: str = "Q4_K_M",
) -> SpeedEstimate:
    """Calculates realistic token generation speed range based on memory bandwidth and execution mode."""
    if not fit.can_run or fit.fit_type == FitType.UNUSABLE:
        return SpeedEstimate(
            estimated_tok_per_sec=0.0,
            speed_range_tok_per_sec=[0, 0],
            speed_display="<1 tok/s (Unusable)",
            speed_confidence="HIGH",
            speed_notes="Model exceeds memory capacity.",
        )

    # 1. Effective active parameters (for MoE, generation bandwidth is dominated by active experts)
    active_params = model.active_parameters if model.active_parameters > 0 else model.parameters
    bits = get_bits_per_weight(quantization)
    bytes_per_param = bits / 8.0
    active_model_bytes = active_params * bytes_per_param

    # 2. Base memory bandwidth
    if fit.fit_type == FitType.UNIFIED_MEMORY:
        # Apple Silicon unified memory
        bandwidth_gbps = 100.0
        if hardware.primary_gpu and hardware.primary_gpu.memory_bandwidth_gbps:
            bandwidth_gbps = hardware.primary_gpu.memory_bandwidth_gbps
        backend_efficiency = 0.85  # Metal kernel efficiency
    elif fit.fit_type in [FitType.FULL_GPU, FitType.MULTI_GPU]:
        bandwidth_gbps = 500.0
        if hardware.primary_gpu and hardware.primary_gpu.memory_bandwidth_gbps:
            bandwidth_gbps = hardware.primary_gpu.memory_bandwidth_gbps
        if fit.fit_type == FitType.MULTI_GPU:
            # Multi-GPU tensor parallelism has slight interconnect overhead
            backend_efficiency = 0.80
        else:
            backend_efficiency = 0.88  # CUDA kernel efficiency
    elif fit.fit_type == FitType.PARTIAL_OFFLOAD:
        # Heavily bottlenecked by PCIe bus bandwidth (~25-30 GB/s) for offloaded layers
        gpu_pct = fit.gpu_layers_pct
        gpu_bw = hardware.primary_gpu.memory_bandwidth_gbps if hardware.primary_gpu else 500.0
        pcie_bw = 25.0  # PCIe 4.0 transfer rate
        # Harmonic mean / weighted latency
        effective_bw = 1.0 / ((gpu_pct / gpu_bw) + ((1.0 - gpu_pct) / pcie_bw))
        bandwidth_gbps = effective_bw
        backend_efficiency = 0.70
    else:  # CPU_ONLY
        # System RAM bandwidth (DDR4 ~30-40 GB/s, DDR5 ~50-80 GB/s)
        bandwidth_gbps = hardware.memory.memory_bandwidth_gbps or 45.0
        backend_efficiency = 0.60  # CPU AVX2/AVX-512 compute limit

    # 3. Calculate theoretical tok/sec
    # tok/sec = bandwidth_bytes_per_sec / active_model_bytes * efficiency
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9
    raw_tok_per_sec = (bandwidth_bytes_per_sec / active_model_bytes) * backend_efficiency

    # Cap realistic limits for interactive text generation
    raw_tok_per_sec = max(0.5, min(raw_tok_per_sec, 180.0))

    # 4. Generate integer range (no false precision)
    low = max(1, int(round(raw_tok_per_sec * 0.88)))
    high = max(low + 1, int(round(raw_tok_per_sec * 1.12)))
    mean_speed = round((low + high) / 2.0, 1)

    # 5. Confidence determination
    if fit.fit_type in [FitType.FULL_GPU, FitType.UNIFIED_MEMORY]:
        conf = "HIGH" if hardware.primary_gpu and hardware.primary_gpu.memory_bandwidth_gbps else "MEDIUM"
        notes = f"Fast generation on {fit.fit_label} (~{bandwidth_gbps:.0f} GB/s bandwidth)."
    elif fit.fit_type == FitType.PARTIAL_OFFLOAD:
        conf = "MEDIUM"
        notes = f"Constrained by PCIe bus transfer rate on {int(fit.gpu_layers_pct * 100)}% GPU offload."
    else:
        conf = "LOW" if raw_tok_per_sec < 4 else "MEDIUM"
        notes = "CPU execution limited by system RAM bandwidth and CPU memory bus."

    display = f"~{low}–{high} tok/s"
    if high < 2:
        display = "<2 tok/s (Slow)"

    return SpeedEstimate(
        estimated_tok_per_sec=mean_speed,
        speed_range_tok_per_sec=[low, high],
        speed_display=display,
        speed_confidence=conf,
        speed_notes=notes,
    )
