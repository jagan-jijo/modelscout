"""Quantization definitions, bits per weight, and quality multipliers."""

from enum import Enum
from typing import Dict


class QuantizationType(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    FP8 = "FP8"
    INT8 = "INT8"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q3_K_M = "Q3_K_M"
    Q2_K = "Q2_K"


# Effective bits per weight for each quantization
BITS_PER_WEIGHT: Dict[str, float] = {
    "FP32": 32.0,
    "FP16": 16.0,
    "BF16": 16.0,
    "FP8": 8.5,
    "INT8": 8.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.54,
    "Q5_K_S": 5.34,
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.3,
    "Q3_K_M": 3.44,
    "Q2_K": 2.63,
}

# Approximate quality degradation penalty factor (1.0 = full quality, <1.0 = minor quality loss)
QUANTIZATION_QUALITY_FACTOR: Dict[str, float] = {
    "FP32": 1.0,
    "FP16": 1.0,
    "BF16": 1.0,
    "FP8": 0.99,
    "INT8": 0.98,
    "Q8_0": 0.995,
    "Q6_K": 0.985,
    "Q5_K_M": 0.97,
    "Q5_K_S": 0.95,
    "Q4_K_M": 0.93,
    "Q4_K_S": 0.90,
    "Q3_K_M": 0.82,
    "Q2_K": 0.68,
}


def get_bits_per_weight(quant: str) -> float:
    """Returns effective bits per weight for quantization format."""
    return BITS_PER_WEIGHT.get(quant.upper(), 4.5)


def get_quant_quality_multiplier(quant: str) -> float:
    """Returns quality retention multiplier for quantization format."""
    return QUANTIZATION_QUALITY_FACTOR.get(quant.upper(), 0.93)
