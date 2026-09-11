"""GPU abstractions and utilities."""

from typing import Optional
from modelscout.hardware.types import GpuInfo


def normalize_gpu_name(raw_name: str) -> str:
    """Normalizes GPU model names for matching."""
    name = raw_name.replace("NVIDIA", "").replace("AMD", "").replace("Intel", "").strip()
    return " ".join(name.split())
