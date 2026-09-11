"""External sources package for Hugging Face, Ollama, and NVIDIA Build."""

from modelscout.sources.huggingface import HuggingFaceModelSummary, search_huggingface_models
from modelscout.sources.nvidia import NvidiaBuildModel, get_nvidia_build_catalog
from modelscout.sources.ollama import LocalOllamaModel, get_local_ollama_models

__all__ = [
    "search_huggingface_models",
    "HuggingFaceModelSummary",
    "get_local_ollama_models",
    "LocalOllamaModel",
    "get_nvidia_build_catalog",
    "NvidiaBuildModel",
]
