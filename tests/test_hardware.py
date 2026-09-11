"""Tests for hardware detection, typing, and AI Hardware scoring."""

import pytest
from modelscout.hardware.detector import compute_hardware_score, detect_runtimes, detect_system_hardware
from modelscout.hardware.types import (
    CpuInfo,
    GpuInfo,
    MemoryInfo,
    RuntimeCapability,
    StorageInfo,
    SystemHardware,
)


def test_detect_system_hardware_live():
    hw = detect_system_hardware()
    assert hw.os_name in ["Darwin", "Linux", "Windows"]
    assert hw.cpu.name != ""
    assert hw.memory.total_ram_gb > 0
    assert hw.storage.total_gb >= 0
    assert hw.hardware_score is not None
    assert 0 <= hw.hardware_score.overall_score <= 100


def test_hardware_simulation_rtx4090():
    hw = detect_system_hardware(
        cpu_override="AMD Ryzen 9 7950X",
        gpu_override="NVIDIA GeForce RTX 4090",
        ram_override_gb=64,
        vram_override_gb=24,
    )
    assert hw.cpu.vendor == "AMD"
    assert "7950X" in hw.cpu.name
    assert len(hw.gpus) == 1
    assert hw.gpus[0].vram_gb == 24.0
    assert hw.gpus[0].cuda_support is True
    assert hw.memory.unified_memory is False
    assert hw.memory.total_ram_gb == 64.0
    assert hw.hardware_score.overall_score >= 85


def test_hardware_simulation_apple_silicon():
    hw = detect_system_hardware(
        cpu_override="Apple M3 Max",
        gpu_override="Apple M3 Max GPU",
        ram_override_gb=128,
    )
    assert hw.is_apple_silicon is True
    assert hw.memory.unified_memory is True
    assert hw.memory.total_ram_gb == 128.0
    assert hw.gpus[0].metal_support is True
    assert hw.hardware_score.overall_score >= 90


def test_hardware_scoring_breakdown():
    cpu = CpuInfo(name="Intel Core i9-14900K", logical_cores=32, features=["AVX2"])
    gpu = GpuInfo(name="RTX 4090", vendor="NVIDIA", vram_gb=24, memory_bandwidth_gbps=1008, cuda_support=True)
    mem = MemoryInfo(total_ram_gb=64, memory_bandwidth_gbps=60)
    rt = RuntimeCapability(cuda_available=True, ollama_installed=True)

    score = compute_hardware_score(cpu, [gpu], mem, rt)
    assert score.overall_score >= 88
    assert score.gpu_score >= 90
    assert score.memory_capacity_score >= 90
    assert score.memory_bandwidth_score >= 90
    assert score.ai_readiness_score >= 80
