"""Linux hardware detection using /proc, sysfs, and vendor tools."""

import os
import re
from typing import List, Tuple
import psutil

from modelscout.hardware.amd import detect_amd_gpus
from modelscout.hardware.intel import detect_intel_gpus
from modelscout.hardware.nvidia import detect_nvidia_gpus
from modelscout.hardware.types import (
    CpuInfo,
    GpuInfo,
    MemoryInfo,
    StorageInfo,
)


def detect_linux_cpu() -> Tuple[CpuInfo, bool, str]:
    """Detects CPU on Linux via /proc/cpuinfo."""
    name = "Linux CPU"
    vendor = "Unknown"
    arch = os.uname().machine
    features: List[str] = []

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("vendor_id"):
                    v = line.split(":", 1)[1].strip()
                    vendor = "Intel" if "Intel" in v else ("AMD" if "AMD" in v else v)
                elif line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip().split()
                    if "avx512f" in flags:
                        features.append("AVX-512")
                    if "avx2" in flags:
                        features.append("AVX2")
                    if "amx_bf16" in flags:
                        features.append("AMX")
    except Exception:
        pass

    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores

    cpu = CpuInfo(
        name=name,
        vendor=vendor,
        architecture=arch,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        features=features,
    )
    return cpu, False, None


def detect_linux_gpus() -> List[GpuInfo]:
    """Detects GPUs on Linux across NVIDIA, AMD, and Intel."""
    gpus: List[GpuInfo] = []
    # 1. NVIDIA
    gpus.extend(detect_nvidia_gpus())
    # 2. AMD
    if not gpus:
        gpus.extend(detect_amd_gpus())
    # 3. Intel
    if not gpus:
        gpus.extend(detect_intel_gpus())
    return gpus


def detect_linux_memory() -> MemoryInfo:
    """Detects RAM on Linux."""
    vm = psutil.virtual_memory()
    total_gb = round(vm.total / (1024**3), 1)
    avail_gb = round(vm.available / (1024**3), 1)

    return MemoryInfo(
        total_ram_bytes=vm.total,
        total_ram_gb=total_gb,
        available_ram_bytes=vm.available,
        available_ram_gb=avail_gb,
        unified_memory=False,
        memory_type="DDR4/DDR5",
        memory_bandwidth_gbps=50.0,  # Average dual channel DDR4/DDR5
    )


def detect_linux_storage() -> StorageInfo:
    """Detects storage on Linux."""
    try:
        du = psutil.disk_usage("/")
        return StorageInfo(
            total_bytes=du.total,
            total_gb=round(du.total / (1024**3), 1),
            available_bytes=du.free,
            available_gb=round(du.free / (1024**3), 1),
            storage_type="SSD",
        )
    except Exception:
        return StorageInfo(storage_type="SSD")
