"""Tests for KV cache, memory estimation, fit types, and speed ranges."""

import pytest
from modelscout.estimation.fit import FitType, evaluate_hardware_fit
from modelscout.estimation.kv_cache import calculate_kv_cache_bytes, estimate_kv_cache_from_params
from modelscout.estimation.memory import estimate_model_memory
from modelscout.estimation.speed import estimate_generation_speed
from modelscout.hardware.types import (
    CpuInfo,
    GpuInfo,
    MemoryInfo,
    RuntimeCapability,
    StorageInfo,
    SystemHardware,
)
from modelscout.models.metadata import ModelMetadata


@pytest.fixture
def sample_8b_model():
    return ModelMetadata(
        id="sample-8b",
        display_name="Sample 8B",
        family="Sample",
        publisher="Test",
        parameters=8_000_000_000,
        active_parameters=8_000_000_000,
        architecture="Dense",
        context_length=131072,
    )


@pytest.fixture
def sample_70b_model():
    return ModelMetadata(
        id="sample-70b",
        display_name="Sample 70B",
        family="Sample",
        publisher="Test",
        parameters=70_000_000_000,
        active_parameters=70_000_000_000,
        architecture="Dense",
        context_length=131072,
    )


def test_kv_cache_calculation():
    # 8k context, 32 layers, 8 kv heads, 128 head dim, FP16 (2 bytes)
    bytes_kv = calculate_kv_cache_bytes(8192, num_layers=32, kv_heads=8, head_dim=128, bytes_per_element=2.0)
    assert bytes_kv == 2 * 32 * 8 * 128 * 8192 * 2  # 1,073,741,824 bytes = 1 GB
    assert abs(bytes_kv / (1024**3) - 1.0) < 0.01


def test_memory_estimation_8b(sample_8b_model):
    mem = estimate_model_memory(sample_8b_model, quantization="Q4_K_M", context_tokens=8192)
    assert 4.0 <= mem.weights_gb <= 5.5
    assert mem.kv_cache_gb > 0.5
    assert 5.5 <= mem.total_required_gb <= 7.5


def test_fit_and_speed_on_rtx_4090(sample_8b_model, sample_70b_model):
    hw = SystemHardware(
        os_name="Linux",
        os_version="6.5",
        architecture="x86_64",
        cpu=CpuInfo(name="Ryzen 9"),
        gpus=[GpuInfo(name="NVIDIA GeForce RTX 4090", vendor="NVIDIA", vram_gb=24.0, memory_bandwidth_gbps=1008.0, cuda_support=True)],
        memory=MemoryInfo(total_ram_gb=64.0, available_ram_gb=50.0),
        storage=StorageInfo(available_gb=500.0),
    )

    # 8B model fits in Full GPU
    mem_8b = estimate_model_memory(sample_8b_model, quantization="Q4_K_M")
    fit_8b = evaluate_hardware_fit(mem_8b, hw)
    assert fit_8b.fit_type == FitType.FULL_GPU
    assert fit_8b.can_run is True

    speed_8b = estimate_generation_speed(sample_8b_model, hw, fit_8b, quantization="Q4_K_M")
    assert speed_8b.estimated_tok_per_sec > 100.0
    assert len(speed_8b.speed_range_tok_per_sec) == 2
    assert speed_8b.speed_range_tok_per_sec[0] < speed_8b.speed_range_tok_per_sec[1]

    # 70B model requires ~45GB, so on 24GB RTX 4090 + 64GB RAM it uses partial offload
    mem_70b = estimate_model_memory(sample_70b_model, quantization="Q4_K_M")
    fit_70b = evaluate_hardware_fit(mem_70b, hw)
    assert fit_70b.fit_type == FitType.PARTIAL_OFFLOAD
    assert fit_70b.can_run is True
    speed_70b = estimate_generation_speed(sample_70b_model, hw, fit_70b, quantization="Q4_K_M")
    assert speed_70b.estimated_tok_per_sec < speed_8b.estimated_tok_per_sec
