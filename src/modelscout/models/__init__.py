"""Models package for discovery, normalization, artifacts, and quantization."""

from modelscout.models.artifacts import ModelArtifact
from modelscout.models.discovery import discover_available_models
from modelscout.models.metadata import ModelMetadata
from modelscout.models.normalization import extract_model_family, normalize_model_name, parse_parameters_str
from modelscout.models.quantization import BITS_PER_WEIGHT, QuantizationType, get_bits_per_weight, get_quant_quality_multiplier

__all__ = [
    "ModelMetadata",
    "ModelArtifact",
    "QuantizationType",
    "BITS_PER_WEIGHT",
    "get_bits_per_weight",
    "get_quant_quality_multiplier",
    "normalize_model_name",
    "extract_model_family",
    "parse_parameters_str",
    "discover_available_models",
]
