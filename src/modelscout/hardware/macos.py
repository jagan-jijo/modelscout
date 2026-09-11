"""macOS hardware detector using sysctl, system_profiler, and sw_vers."""

import re
import shutil
import subprocess
from typing import Optional, Tuple
import psutil

from modelscout.hardware.types import (
    CpuInfo,
    GpuInfo,
    MemoryInfo,
    StorageInfo,
)


def _run_cmd(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return res.stdout.strip()
    except Exception:
        return ""


def detect_macos_cpu() -> Tuple[CpuInfo, bool, Optional[str]]:
    """Detects CPU on macOS. Returns (CpuInfo, is_apple_silicon, generation)."""
    brand = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    ncpu_str = _run_cmd(["sysctl", "-n", "hw.ncpu"])
    logical_cores = int(ncpu_str) if ncpu_str.isdigit() else psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores

    is_apple_silicon = False
    apple_gen = None
    vendor = "Apple" if "Apple" in brand else ("Intel" if "Intel" in brand else "Unknown")

    # Match Apple Silicon generically (M1, M2, M3, M4, M5..., Pro, Max, Ultra)
    apple_match = re.search(r"Apple\s+(M\d+)(\s+(Pro|Max|Ultra))?", brand, re.IGNORECASE)
    if apple_match:
        is_apple_silicon = True
        apple_gen = apple_match.group(1).upper()
        arch = "arm64"
    else:
        # Fallback check arch
        arch = _run_cmd(["uname", "-m"])
        if arch == "arm64" or "Apple" in brand:
            is_apple_silicon = True
            m = re.search(r"M\d+", brand)
            if m:
                apple_gen = m.group(0).upper()

    features = []
    if is_apple_silicon:
        features.extend(["NEON", "Apple AMX", "Unified Memory"])
    else:
        features.extend(["AVX2"])

    cpu = CpuInfo(
        name=brand or "Apple Silicon CPU",
        vendor=vendor,
        architecture="Apple Silicon" if is_apple_silicon else "x86_64",
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        features=features,
    )
    return cpu, is_apple_silicon, apple_gen


def detect_macos_gpu(is_apple_silicon: bool, total_ram_gb: float, cpu_name: str) -> list[GpuInfo]:
    """Detects GPU(s) on macOS."""
    gpus: list[GpuInfo] = []

    if is_apple_silicon:
        # Apple Silicon unified GPU
        # Check system_profiler for display data
        sp_disp = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        cores_match = re.search(r"Total Number of Cores:\s*(\d+)", sp_disp)
        gpu_cores = int(cores_match.group(1)) if cores_match else 0

        # Memory bandwidth estimation based on Apple Silicon gen
        bandwidth = 100.0  # baseline
        name = f"{cpu_name} GPU" if not cpu_name.endswith("GPU") else cpu_name
        if "Max" in cpu_name:
            bandwidth = 400.0 if "M4" not in cpu_name else 546.0
        elif "Ultra" in cpu_name:
            bandwidth = 800.0
        elif "Pro" in cpu_name:
            bandwidth = 150.0 if "M4" not in cpu_name else 273.0
        elif "M4" in cpu_name:
            bandwidth = 120.0
        elif "M3" in cpu_name or "M2" in cpu_name:
            bandwidth = 100.0
        elif "M1" in cpu_name:
            bandwidth = 68.25

        # Metal is natively supported on all Apple Silicon
        has_metal = shutil.which("metal") is not None or True

        gpu = GpuInfo(
            name=name,
            vendor="Apple",
            architecture="Apple Silicon GPU",
            vram_bytes=int(total_ram_gb * 1024 * 1024 * 1024 * 0.75),  # 75% usable for VRAM
            vram_gb=round(total_ram_gb * 0.75, 1),
            memory_bandwidth_gbps=bandwidth,
            unified_memory=True,
            integrated=True,
            metal_support=has_metal,
            cuda_support=False,
            rocm_support=False,
            vulkan_support=False,
        )
        gpus.append(gpu)
    else:
        # Intel Mac discrete/integrated GPU
        sp_disp = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        chipset_match = re.search(r"Chipset Model:\s*(.+)", sp_disp)
        vram_match = re.search(r"VRAM \(Total\):\s*(\d+)\s*(MB|GB)", sp_disp)
        name = chipset_match.group(1).strip() if chipset_match else "Intel Graphics"

        vram_gb = 1.5
        if vram_match:
            val = float(vram_match.group(1))
            unit = vram_match.group(2)
            vram_gb = val if unit == "GB" else round(val / 1024.0, 1)

        vendor = "Intel"
        if "AMD" in name or "Radeon" in name:
            vendor = "AMD"
        elif "NVIDIA" in name or "GeForce" in name:
            vendor = "NVIDIA"

        gpu = GpuInfo(
            name=name,
            vendor=vendor,
            architecture="Discrete/Integrated",
            vram_bytes=int(vram_gb * 1024 * 1024 * 1024),
            vram_gb=vram_gb,
            unified_memory=False,
            integrated=(vendor == "Intel"),
            metal_support=True,
        )
        gpus.append(gpu)

    return gpus


def detect_macos_memory(is_apple_silicon: bool) -> MemoryInfo:
    """Detects RAM on macOS."""
    vm = psutil.virtual_memory()
    total_gb = round(vm.total / (1024**3), 1)
    avail_gb = round(vm.available / (1024**3), 1)

    return MemoryInfo(
        total_ram_bytes=vm.total,
        total_ram_gb=total_gb,
        available_ram_bytes=vm.available,
        available_ram_gb=avail_gb,
        unified_memory=is_apple_silicon,
        memory_type="Unified LPDDR5/LPDDR5X" if is_apple_silicon else "DDR4",
    )


def detect_macos_storage() -> StorageInfo:
    """Detects main disk storage on macOS."""
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
