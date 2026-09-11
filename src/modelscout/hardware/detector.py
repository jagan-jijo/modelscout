"""Master hardware detection and AI Hardware scoring engine."""

import platform
import re
import shutil
import sys
from typing import List, Optional, Union
import httpx

from modelscout.hardware.linux import (
    detect_linux_cpu,
    detect_linux_gpus,
    detect_linux_memory,
    detect_linux_storage,
)
from modelscout.hardware.macos import (
    detect_macos_cpu,
    detect_macos_gpu,
    detect_macos_memory,
    detect_macos_storage,
)
from modelscout.hardware.types import (
    CpuInfo,
    GpuInfo,
    HardwareScoreBreakdown,
    MemoryInfo,
    RuntimeCapability,
    StorageInfo,
    SystemHardware,
)


def detect_runtimes() -> RuntimeCapability:
    """Detects installed and running local AI inference runtimes."""
    ollama_installed = shutil.which("ollama") is not None
    ollama_running = False
    if ollama_installed:
        try:
            r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=0.8)
            if r.status_code == 200:
                ollama_running = True
        except Exception:
            ollama_running = False

    llama_cpp_installed = (
        shutil.which("llama-cli") is not None
        or shutil.which("llama-server") is not None
        or shutil.which("llama-bench") is not None
    )

    is_mac = platform.system() == "Darwin"
    metal_available = is_mac and platform.machine() == "arm64"
    cuda_available = shutil.which("nvidia-smi") is not None or shutil.which("nvcc") is not None
    rocm_available = shutil.which("rocm-smi") is not None or shutil.which("rocminfo") is not None
    vulkan_available = shutil.which("vulkaninfo") is not None

    pytorch_version = None
    try:
        import torch  # type: ignore
        pytorch_version = torch.__version__
    except Exception:
        pass

    return RuntimeCapability(
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        llama_cpp_installed=llama_cpp_installed,
        metal_available=metal_available,
        cuda_available=cuda_available,
        rocm_available=rocm_available,
        vulkan_available=vulkan_available,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        pytorch_version=pytorch_version,
    )


def compute_hardware_score(
    cpu: CpuInfo,
    gpus: list[GpuInfo],
    memory: MemoryInfo,
    runtimes: RuntimeCapability,
) -> HardwareScoreBreakdown:
    """Calculates explainable 0-100 AI Hardware Capability Score and breakdown."""
    # 1. GPU Score (0-100)
    primary_gpu = max(gpus, key=lambda g: g.vram_gb) if gpus else None
    gpu_score = 15
    if primary_gpu:
        vram = primary_gpu.vram_gb
        if primary_gpu.unified_memory:
            # Unified memory on Apple Silicon
            if vram >= 96:
                gpu_score = 95
            elif vram >= 64:
                gpu_score = 90
            elif vram >= 36:
                gpu_score = 85
            elif vram >= 24:
                gpu_score = 78
            elif vram >= 16:
                gpu_score = 70
            elif vram >= 8:
                gpu_score = 55
            else:
                gpu_score = 40
        else:
            # Discrete GPU (NVIDIA / AMD)
            if vram >= 32:
                gpu_score = 98
            elif vram >= 24:
                gpu_score = 94
            elif vram >= 16:
                gpu_score = 85
            elif vram >= 12:
                gpu_score = 75
            elif vram >= 8:
                gpu_score = 60
            elif vram >= 6:
                gpu_score = 45
            else:
                gpu_score = 30

        if primary_gpu.cuda_support or primary_gpu.metal_support:
            gpu_score = min(100, gpu_score + 5)

    # 2. Memory Capacity Score (0-100)
    ram = memory.total_ram_gb
    if ram >= 128:
        memory_cap_score = 98
    elif ram >= 64:
        memory_cap_score = 92
    elif ram >= 36:
        memory_cap_score = 85
    elif ram >= 32:
        memory_cap_score = 80
    elif ram >= 16:
        memory_cap_score = 65
    elif ram >= 8:
        memory_cap_score = 45
    else:
        memory_cap_score = 25

    # 3. Memory Bandwidth Score (0-100)
    bandwidth = 50.0
    if primary_gpu and primary_gpu.memory_bandwidth_gbps:
        bandwidth = primary_gpu.memory_bandwidth_gbps
    elif memory.memory_bandwidth_gbps:
        bandwidth = memory.memory_bandwidth_gbps

    if bandwidth >= 1500:
        bandwidth_score = 99
    elif bandwidth >= 900:
        bandwidth_score = 95
    elif bandwidth >= 500:
        bandwidth_score = 88
    elif bandwidth >= 300:
        bandwidth_score = 80
    elif bandwidth >= 150:
        bandwidth_score = 72
    elif bandwidth >= 90:
        bandwidth_score = 60
    else:
        bandwidth_score = 45

    # 4. CPU Score (0-100)
    cores = cpu.logical_cores
    cpu_score = 50
    if cores >= 32:
        cpu_score = 95
    elif cores >= 16:
        cpu_score = 88
    elif cores >= 12:
        cpu_score = 80
    elif cores >= 8:
        cpu_score = 72
    elif cores >= 4:
        cpu_score = 55
    else:
        cpu_score = 40

    if any(feat in cpu.features for feat in ["NEON", "AVX2", "AVX-512"]):
        cpu_score = min(100, cpu_score + 5)

    # 5. AI Readiness Score (0-100)
    readiness = 50
    if runtimes.metal_available or runtimes.cuda_available:
        readiness += 25
    if runtimes.ollama_installed:
        readiness += 15
    if runtimes.llama_cpp_installed:
        readiness += 10
    ai_readiness_score = min(100, readiness)

    # Weighted overall score
    overall = int(
        (gpu_score * 0.40)
        + (memory_cap_score * 0.25)
        + (bandwidth_score * 0.20)
        + (cpu_score * 0.10)
        + (ai_readiness_score * 0.05)
    )
    overall = max(10, min(100, overall))

    summary = (
        f"AI Hardware Score: {overall}/100 — "
        f"{'Exceptional' if overall >= 90 else 'Great' if overall >= 80 else 'Capable' if overall >= 65 else 'Entry-level'} "
        f"local AI inference configuration."
    )

    return HardwareScoreBreakdown(
        overall_score=overall,
        gpu_score=gpu_score,
        memory_capacity_score=memory_cap_score,
        memory_bandwidth_score=bandwidth_score,
        cpu_score=cpu_score,
        ai_readiness_score=ai_readiness_score,
        summary=summary,
    )


def detect_system_hardware(
    cpu_override: Optional[str] = None,
    gpu_override: Optional[Union[str, List[str]]] = None,
    ram_override_gb: Optional[float] = None,
    vram_override_gb: Optional[float] = None,
) -> SystemHardware:
    """Master hardware detector with simulation overrides and graceful fallbacks."""
    os_name = platform.system()
    os_version = platform.release()
    arch = platform.machine()

    if os_name == "Darwin":
        cpu, is_apple_silicon, apple_gen = detect_macos_cpu()
        memory = detect_macos_memory(is_apple_silicon)
        gpus = detect_macos_gpu(is_apple_silicon, memory.total_ram_gb, cpu.name)
        storage = detect_macos_storage()
    elif os_name == "Linux":
        cpu, is_apple_silicon, apple_gen = detect_linux_cpu()
        gpus = detect_linux_gpus()
        memory = detect_linux_memory()
        storage = detect_linux_storage()
    else:
        # Generic fallback
        cpu = CpuInfo(name=f"{platform.processor() or 'Generic CPU'}", architecture=arch)
        memory = MemoryInfo(total_ram_gb=16.0, available_ram_gb=12.0)
        gpus = []
        storage = StorageInfo()
        is_apple_silicon = False
        apple_gen = None

    # Handle simulation overrides
    if cpu_override:
        cpu.name = cpu_override
        if "Ryzen" in cpu_override or "AMD" in cpu_override:
            cpu.vendor = "AMD"
        elif "Intel" in cpu_override or "Core" in cpu_override:
            cpu.vendor = "Intel"
        elif "Apple" in cpu_override or "M1" in cpu_override or "M2" in cpu_override or "M3" in cpu_override or "M4" in cpu_override:
            cpu.vendor = "Apple"
            is_apple_silicon = True

    if ram_override_gb is not None and ram_override_gb > 0:
        memory.total_ram_gb = float(ram_override_gb)
        memory.total_ram_bytes = int(ram_override_gb * 1024 * 1024 * 1024)
        memory.available_ram_gb = round(ram_override_gb * 0.85, 1)
        memory.available_ram_bytes = int(memory.available_ram_gb * 1024 * 1024 * 1024)

    if gpu_override:
        # Normalize gpu_override to list of string tokens
        raw_items: list[str] = []
        if isinstance(gpu_override, list):
            for item in gpu_override:
                raw_items.extend(p.strip() for p in item.split(",") if p.strip())
        elif isinstance(gpu_override, str):
            raw_items.extend(p.strip() for p in gpu_override.split(",") if p.strip())

        expanded_gpu_names: list[str] = []
        for raw in raw_items:
            multiplier = 1
            m_match = re.match(r"^(\d+)\s*[xX*]\s*(.+)$", raw)
            if m_match:
                multiplier = int(m_match.group(1))
                clean_name = m_match.group(2).strip()
            else:
                clean_name = raw
            for _ in range(max(1, multiplier)):
                expanded_gpu_names.append(clean_name)

        sim_gpus: list[GpuInfo] = []
        for idx, gname in enumerate(expanded_gpu_names, 1):
            is_apple = any(k in gname for k in ["Apple", "M1", "M2", "M3", "M4", "M5"])
            if is_apple:
                is_apple_silicon = True
                memory.unified_memory = True
                vram = vram_override_gb or round(memory.total_ram_gb * 0.75, 1)
                if "Ultra" in gname:
                    bandwidth = 800.0
                elif "Max" in gname:
                    bandwidth = 546.0 if "M4" in gname else 400.0
                elif "Pro" in gname:
                    bandwidth = 273.0 if "M4" in gname else 150.0
                elif "M4" in gname:
                    bandwidth = 120.0
                elif "M1" in gname:
                    bandwidth = 68.25
                else:
                    bandwidth = 100.0
                arch_name = "Apple Silicon GPU"
                vendor_name = "Apple"
                if cpu.logical_cores < 14 and ("Max" in cpu.name or "Pro" in cpu.name):
                    cpu.logical_cores = 16 if "Max" in cpu.name else 14
                    cpu.physical_cores = cpu.logical_cores
            else:
                is_apple_silicon = False
                memory.unified_memory = False
                # Known GPU specs lookup
                g_upper = gname.upper()
                if "5090" in g_upper:
                    vram = 32.0; bandwidth = 1792.0; arch_name = "Blackwell"; vendor_name = "NVIDIA"
                elif "5080" in g_upper:
                    vram = 16.0; bandwidth = 1024.0; arch_name = "Blackwell"; vendor_name = "NVIDIA"
                elif "5070" in g_upper:
                    vram = 16.0 if "TI" in g_upper else 12.0; bandwidth = 896.0 if "TI" in g_upper else 672.0; arch_name = "Blackwell"; vendor_name = "NVIDIA"
                elif "4090" in g_upper:
                    vram = 24.0; bandwidth = 1008.0; arch_name = "Ada Lovelace"; vendor_name = "NVIDIA"
                elif "4080" in g_upper:
                    vram = 16.0; bandwidth = 736.0; arch_name = "Ada Lovelace"; vendor_name = "NVIDIA"
                elif "4070" in g_upper:
                    vram = 16.0 if "TI" in g_upper else 12.0; bandwidth = 672.0 if "TI" in g_upper else 504.0; arch_name = "Ada Lovelace"; vendor_name = "NVIDIA"
                elif "3090" in g_upper:
                    vram = 24.0; bandwidth = 936.0; arch_name = "Ampere"; vendor_name = "NVIDIA"
                elif "3080" in g_upper:
                    vram = 10.0; bandwidth = 760.0; arch_name = "Ampere"; vendor_name = "NVIDIA"
                elif "H100" in g_upper or "H200" in g_upper:
                    vram = 141.0 if "H200" in g_upper else 80.0; bandwidth = 4800.0 if "H200" in g_upper else 3350.0; arch_name = "Hopper"; vendor_name = "NVIDIA"
                elif "B200" in g_upper:
                    vram = 180.0; bandwidth = 8000.0; arch_name = "Blackwell"; vendor_name = "NVIDIA"
                elif "A100" in g_upper:
                    vram = 80.0 if "80" in g_upper else 40.0; bandwidth = 2039.0 if "80" in g_upper else 1555.0; arch_name = "Ampere"; vendor_name = "NVIDIA"
                elif "7900" in g_upper:
                    vram = 24.0 if "XTX" in g_upper else 20.0; bandwidth = 960.0 if "XTX" in g_upper else 800.0; arch_name = "RDNA 3"; vendor_name = "AMD"
                elif "7800" in g_upper:
                    vram = 16.0; bandwidth = 624.0; arch_name = "RDNA 3"; vendor_name = "AMD"
                elif "9070" in g_upper:
                    vram = 16.0; bandwidth = 640.0; arch_name = "RDNA 4"; vendor_name = "AMD"
                elif "MI300" in g_upper:
                    vram = 192.0; bandwidth = 5300.0; arch_name = "CDNA 3"; vendor_name = "AMD"
                elif "B580" in g_upper:
                    vram = 12.0; bandwidth = 456.0; arch_name = "Battlemage"; vendor_name = "Intel"
                elif "A770" in g_upper:
                    vram = 16.0; bandwidth = 560.0; arch_name = "Xe HPG"; vendor_name = "Intel"
                else:
                    vram = 24.0; bandwidth = 500.0; arch_name = "Discrete GPU"
                    vendor_name = "NVIDIA" if any(k in g_upper for k in ["RTX", "GTX", "GEFORCE", "TITAN"]) else ("AMD" if "RADEON" in g_upper else "Discrete")

                if vram_override_gb is not None:
                    vram = float(vram_override_gb)

            disp_name = gname if len(expanded_gpu_names) == 1 else f"{gname} #{idx}"
            sim_gpus.append(
                GpuInfo(
                    name=disp_name,
                    vendor=vendor_name,
                    architecture=arch_name,
                    vram_bytes=int(vram * 1024 * 1024 * 1024),
                    vram_gb=float(vram),
                    memory_bandwidth_gbps=bandwidth,
                    device_count=1,
                    cuda_support=vendor_name == "NVIDIA",
                    rocm_support=vendor_name == "AMD",
                    metal_support=is_apple,
                    unified_memory=is_apple,
                )
            )
        gpus = sim_gpus
    elif vram_override_gb is not None and gpus:
        for g in gpus:
            g.vram_gb = float(vram_override_gb)
            g.vram_bytes = int(vram_override_gb * 1024 * 1024 * 1024)

    runtimes = detect_runtimes()
    hw_score = compute_hardware_score(cpu, gpus, memory, runtimes)

    return SystemHardware(
        os_name=os_name,
        os_version=os_version,
        architecture=arch,
        cpu=cpu,
        gpus=gpus,
        memory=memory,
        storage=storage,
        runtimes=runtimes,
        is_apple_silicon=is_apple_silicon,
        apple_silicon_generation=apple_gen,
        hardware_score=hw_score,
    )


from modelscout.hardware.amd import detect_amd_gpus
from modelscout.hardware.apple import detect_apple_gpu, detect_apple_gpu_linux
from modelscout.hardware.cpu import detect_avx_support, detect_cpu_cores, detect_cpu_name
from modelscout.hardware.intel import detect_intel_gpus
from modelscout.hardware.memory import detect_disk_free_bytes, detect_ram_bytes
from modelscout.hardware.nvidia import detect_nvidia_gpus
from modelscout.hardware.types import HardwareInfo
from modelscout.hardware.windows import detect_windows_gpus


def detect_hardware() -> HardwareInfo:
    """Detect all hardware. Each detector is fail-safe (returns empty on error)."""
    os_name = platform.system().lower()
    if os_name not in ("linux", "darwin", "windows"):
        os_name = "linux"

    # GPU detection
    gpus = []
    gpus.extend(detect_nvidia_gpus())
    if os_name == "linux":
        gpus.extend(detect_amd_gpus())
        gpus.extend(detect_intel_gpus())
        gpus.extend(detect_apple_gpu_linux())
    if os_name == "darwin":
        gpus.extend(detect_apple_gpu())
    if os_name == "windows":
        gpus.extend(detect_windows_gpus())

    # CPU
    cpu_name = detect_cpu_name()
    cpu_cores = detect_cpu_cores()
    has_avx2, has_avx512 = detect_avx_support()

    # Memory
    ram_bytes = detect_ram_bytes()
    disk_free = detect_disk_free_bytes()

    return HardwareInfo(
        gpus=gpus,
        cpu_name=cpu_name or "Unknown CPU",
        cpu_cores=cpu_cores or 1,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        ram_bytes=ram_bytes,
        disk_free_bytes=disk_free,
        os=os_name,
    )
