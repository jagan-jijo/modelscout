"""Hardware module for detection, normalization, and scoring."""

from modelscout.hardware.detector import compute_hardware_score, detect_runtimes, detect_system_hardware
from modelscout.hardware.types import (
    CpuInfo,
    GpuInfo,
    HardwareScoreBreakdown,
    MemoryInfo,
    RuntimeCapability,
    StorageInfo,
    SystemHardware,
)

__all__ = [
    "CpuInfo",
    "GpuInfo",
    "MemoryInfo",
    "StorageInfo",
    "RuntimeCapability",
    "HardwareScoreBreakdown",
    "SystemHardware",
    "detect_system_hardware",
    "compute_hardware_score",
    "detect_runtimes",
]
