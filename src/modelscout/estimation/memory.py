"""Architecture-aware memory estimation for local model execution."""

from typing import Optional
from pydantic import BaseModel
from modelscout.estimation.kv_cache import estimate_kv_cache_from_params
from modelscout.models.artifacts import ModelArtifact
from modelscout.models.metadata import ModelMetadata
from modelscout.models.quantization import get_bits_per_weight


class MemoryRequirement(BaseModel):
    weights_bytes: int
    weights_gb: float
    kv_cache_bytes: int
    kv_cache_gb: float
    activations_bytes: int
    activations_gb: float
    runtime_overhead_bytes: int
    runtime_overhead_gb: float
    total_required_bytes: int
    total_required_gb: float
    recommended_vram_gb: float


def estimate_model_memory(
    model: ModelMetadata,
    quantization: str = "Q4_K_M",
    context_tokens: Optional[int] = None,
    artifact: Optional[ModelArtifact] = None,
) -> MemoryRequirement:
    """Estimates memory requirements: Weights + KV Cache + Activations + Runtime Overhead."""
    target_context = context_tokens or min(model.context_length, 8192)  # Default evaluate at 8k context for standard interactive usage

    # 1. Weights memory
    if artifact and artifact.file_size_bytes > 0:
        weights_bytes = artifact.file_size_bytes
    else:
        bits = get_bits_per_weight(quantization)
        weights_bytes = int(model.parameters * bits / 8.0)

    # 2. KV Cache memory
    kv_bytes = estimate_kv_cache_from_params(
        parameters=model.parameters,
        context_tokens=target_context,
        is_gqa=True,
    )

    # 3. Activation memory (scales with context & hidden dim, ~150MB - 800MB)
    if model.parameters >= 60_000_000_000:
        act_bytes = int(800 * 1024 * 1024)
    elif model.parameters >= 25_000_000_000:
        act_bytes = int(500 * 1024 * 1024)
    elif model.parameters >= 10_000_000_000:
        act_bytes = int(350 * 1024 * 1024)
    else:
        act_bytes = int(200 * 1024 * 1024)

    # 4. Runtime overhead (Metal/CUDA context, driver buffers, scratchpad)
    overhead_bytes = int(512 * 1024 * 1024)  # 512 MB

    total_bytes = weights_bytes + kv_bytes + act_bytes + overhead_bytes

    w_gb = round(weights_bytes / (1024**3), 2)
    kv_gb = round(kv_bytes / (1024**3), 2)
    act_gb = round(act_bytes / (1024**3), 2)
    ov_gb = round(overhead_bytes / (1024**3), 2)
    tot_gb = round(total_bytes / (1024**3), 2)

    # Safety margin for stable generation
    recommended_vram = round(tot_gb * 1.05, 1)

    return MemoryRequirement(
        weights_bytes=weights_bytes,
        weights_gb=w_gb,
        kv_cache_bytes=kv_bytes,
        kv_cache_gb=kv_gb,
        activations_bytes=act_bytes,
        activations_gb=act_gb,
        runtime_overhead_bytes=overhead_bytes,
        runtime_overhead_gb=ov_gb,
        total_required_bytes=total_bytes,
        total_required_gb=tot_gb,
        recommended_vram_gb=recommended_vram,
    )
