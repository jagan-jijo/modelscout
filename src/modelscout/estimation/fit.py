"""Compatibility analysis and fit categorization."""

from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel

from modelscout.estimation.memory import MemoryRequirement
from modelscout.hardware.types import SystemHardware


class FitType(str, Enum):
    FULL_GPU = "FULL_GPU"
    PARTIAL_OFFLOAD = "PARTIAL_OFFLOAD"
    UNIFIED_MEMORY = "UNIFIED_MEMORY"
    CPU_ONLY = "CPU_ONLY"
    MULTI_GPU = "MULTI_GPU"
    UNUSABLE = "UNUSABLE"


class FitEvaluation(BaseModel):
    fit_type: FitType
    can_run: bool
    fits_full_gpu: bool
    gpu_layers_pct: float  # 0.0 to 1.0
    vram_used_gb: float
    ram_used_gb: float
    fit_label: str
    fit_description: str


def evaluate_hardware_fit(
    mem_req: MemoryRequirement,
    hardware: SystemHardware,
    vram_headroom_gb: float = 0.0,
    ram_budget_gb: Optional[float] = None,
) -> FitEvaluation:
    """Evaluates how comfortably the model fits into system GPU and RAM, respecting headroom and budget."""
    req_gb = mem_req.total_required_gb
    total_ram = hardware.memory.total_ram_gb
    if ram_budget_gb is not None and ram_budget_gb > 0:
        total_ram = min(total_ram, float(ram_budget_gb))

    # Case 1: Apple Silicon Unified Memory
    if hardware.is_apple_silicon or hardware.memory.unified_memory:
        safe_usable = max(0.0, total_ram * 0.78 - vram_headroom_gb)  # macOS reserves ~20-22% for OS and windows
        if req_gb <= safe_usable:
            return FitEvaluation(
                fit_type=FitType.UNIFIED_MEMORY,
                can_run=True,
                fits_full_gpu=True,
                gpu_layers_pct=1.0,
                vram_used_gb=req_gb,
                ram_used_gb=req_gb,
                fit_label="Unified Memory (Full Fit)",
                fit_description=f"Fits comfortably in {total_ram} GB unified memory (requires {req_gb} GB).",
            )
        elif req_gb <= total_ram * 0.92:
            return FitEvaluation(
                fit_type=FitType.UNIFIED_MEMORY,
                can_run=True,
                fits_full_gpu=False,
                gpu_layers_pct=0.9,
                vram_used_gb=safe_usable,
                ram_used_gb=req_gb,
                fit_label="Tight Fit (Unified Memory)",
                fit_description=f"Fits with tight memory margins ({req_gb} GB needed out of {total_ram} GB).",
            )
        else:
            return FitEvaluation(
                fit_type=FitType.UNUSABLE,
                can_run=False,
                fits_full_gpu=False,
                gpu_layers_pct=0.0,
                vram_used_gb=0.0,
                ram_used_gb=req_gb,
                fit_label="Exceeds Memory",
                fit_description=f"Requires {req_gb} GB, which exceeds available {total_ram} GB unified memory.",
            )

    # Case 2: Discrete GPU(s)
    raw_vram = sum(g.vram_gb * max(1, g.device_count) for g in hardware.gpus) if hardware.gpus else 0.0
    total_vram = max(0.0, raw_vram - vram_headroom_gb)
    num_gpus = sum(max(1, g.device_count) for g in hardware.gpus) if hardware.gpus else 0

    if total_vram > 0:
        # Check Multi-GPU
        if num_gpus > 1 and req_gb <= total_vram * 0.95:
            return FitEvaluation(
                fit_type=FitType.MULTI_GPU,
                can_run=True,
                fits_full_gpu=True,
                gpu_layers_pct=1.0,
                vram_used_gb=req_gb,
                ram_used_gb=0.5,
                fit_label=f"Multi-GPU ({num_gpus}x GPUs)",
                fit_description=f"Distributed across {num_gpus} GPUs ({req_gb} GB total VRAM used).",
            )

        # Single GPU full fit
        raw_primary = hardware.primary_gpu.vram_gb if hardware.primary_gpu else 0.0
        primary_vram = max(0.0, raw_primary - vram_headroom_gb)
        if req_gb <= primary_vram * 0.95:
            return FitEvaluation(
                fit_type=FitType.FULL_GPU,
                can_run=True,
                fits_full_gpu=True,
                gpu_layers_pct=1.0,
                vram_used_gb=req_gb,
                ram_used_gb=0.5,
                fit_label="Full GPU",
                fit_description=f"Fits completely in VRAM ({req_gb} GB used of {primary_vram} GB).",
            )

        # Partial GPU offload
        if req_gb <= (primary_vram + total_ram * 0.85):
            gpu_pct = min(0.90, max(0.15, primary_vram / req_gb)) if primary_vram > 0 else 0.0
            vram_part = round(primary_vram * 0.90, 1)
            ram_part = round(req_gb - vram_part, 1)
            return FitEvaluation(
                fit_type=FitType.PARTIAL_OFFLOAD,
                can_run=True,
                fits_full_gpu=False,
                gpu_layers_pct=round(gpu_pct, 2),
                vram_used_gb=vram_part,
                ram_used_gb=ram_part,
                fit_label="Partial Offload",
                fit_description=f"Offloaded across GPU ({vram_part} GB VRAM) and system RAM ({ram_part} GB).",
            )

    # Case 3: CPU Only
    if req_gb <= total_ram * 0.85:
        return FitEvaluation(
            fit_type=FitType.CPU_ONLY,
            can_run=True,
            fits_full_gpu=False,
            gpu_layers_pct=0.0,
            vram_used_gb=0.0,
            ram_used_gb=req_gb,
            fit_label="CPU Only",
            fit_description=f"Runs entirely in CPU RAM ({req_gb} GB needed of {total_ram} GB).",
        )

    # Unusable
    return FitEvaluation(
        fit_type=FitType.UNUSABLE,
        can_run=False,
        fits_full_gpu=False,
        gpu_layers_pct=0.0,
        vram_used_gb=0.0,
        ram_used_gb=req_gb,
        fit_label="Exceeds Hardware",
        fit_description=f"Requires {req_gb} GB, which exceeds total system memory.",
    )
