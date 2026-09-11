"""Tests for model parameter parsing, normalization, artifacts, and quantization."""

import pytest
from modelscout.models.artifacts import ModelArtifact
from modelscout.models.normalization import (
    extract_model_family,
    normalize_model_name,
    parse_moe_parameters,
    parse_parameters_str,
)
from modelscout.models.quantization import (
    BITS_PER_WEIGHT,
    get_bits_per_weight,
    get_quant_quality_multiplier,
)


def test_parse_parameters_str():
    assert parse_parameters_str("0.6B") == 600_000_000
    assert parse_parameters_str("1.5B") == 1_500_000_000
    assert parse_parameters_str("7B") == 7_000_000_000
    assert parse_parameters_str("14B") == 14_000_000_000
    assert parse_parameters_str("70B") == 70_000_000_000
    assert parse_parameters_str(None) is None


def test_parse_moe_parameters():
    total, active = parse_moe_parameters("30B-A3B")
    assert total == 30_000_000_000
    assert active == 3_000_000_000

    total2, active2 = parse_moe_parameters("235b-a22b")
    assert total2 == 235_000_000_000
    assert active2 == 22_000_000_000


def test_normalize_model_name_and_family():
    assert normalize_model_name("meta-llama/Llama-3.3-70B-Instruct") == "Llama-3.3-70B"
    assert normalize_model_name("bartowski/Qwen2.5-Coder-32B-Instruct-GGUF") == "Qwen2.5-Coder-32B"
    assert extract_model_family("Qwen3-30B-A3B") == "Qwen3"
    assert extract_model_family("DeepSeek-R1-Distill-Qwen-14B") == "DeepSeek-R1"
    assert extract_model_family("gemma-4-31b-it") == "Gemma 4"


def test_quantization_and_artifact_estimation():
    assert get_bits_per_weight("Q4_K_M") == 4.5
    assert get_bits_per_weight("Q8_0") == 8.5
    assert get_quant_quality_multiplier("Q4_K_M") < 1.0
    assert get_quant_quality_multiplier("FP16") == 1.0

    art_q4 = ModelArtifact.estimate(8_000_000_000, "Q4_K_M")
    assert 4.0 <= art_q4.file_size_gb <= 5.5

    art_q8 = ModelArtifact.estimate(8_000_000_000, "Q8_0")
    assert 8.0 <= art_q8.file_size_gb <= 9.5
