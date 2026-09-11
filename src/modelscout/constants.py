"""Compatibility shim: curated registries now live under ``modelscout.data``.

This module re-exports the same names so existing imports
(``from modelscout.constants import ...``) keep working. New code should import
from the specific ``modelscout.data.*`` submodule instead.
"""

from modelscout.data.framework import (
    FRAMEWORK_OVERHEAD_BYTES,
    MIN_COMPUTE_CAPABILITY_OLLAMA,
    MIN_COMPUTE_CAPABILITY_VLLM,
)
from modelscout.data.gpu import (
    _GiB,
    AMD_SHARED_MEMORY_APU_MARKERS,
    CURATED_GPU_SPECS,
    CuratedGPUSpec,
    GPU_BANDWIDTH,
    GPU_MEMORY_CLOCK_VARIANTS,
    INTEL_PCI_DEVICE_NAMES,
    NVIDIA_COMPUTE_CAPABILITY,
    VULKAN_ONLY_GPUS,
)
from modelscout.data.lineage import (
    MODEL_GENERATION_BONUS_MAX,
    MODEL_GENERATION_PENALTY_MAX,
    MODEL_LINEAGE_VERSIONS,
)
from modelscout.data.quantization import (
    QUANT_BYTES_PER_WEIGHT,
    QUANT_PREFERENCE_ORDER,
    QUANT_QUALITY_PENALTY,
)

__all__ = [
    "_GiB",
    "AMD_SHARED_MEMORY_APU_MARKERS",
    "CURATED_GPU_SPECS",
    "CuratedGPUSpec",
    "FRAMEWORK_OVERHEAD_BYTES",
    "GPU_BANDWIDTH",
    "GPU_MEMORY_CLOCK_VARIANTS",
    "INTEL_PCI_DEVICE_NAMES",
    "MIN_COMPUTE_CAPABILITY_OLLAMA",
    "MIN_COMPUTE_CAPABILITY_VLLM",
    "MODEL_GENERATION_BONUS_MAX",
    "MODEL_GENERATION_PENALTY_MAX",
    "MODEL_LINEAGE_VERSIONS",
    "NVIDIA_COMPUTE_CAPABILITY",
    "QUANT_BYTES_PER_WEIGHT",
    "QUANT_PREFERENCE_ORDER",
    "QUANT_QUALITY_PENALTY",
    "VULKAN_ONLY_GPUS",
]
