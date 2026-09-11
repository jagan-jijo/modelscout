"""NVIDIA Build and NIM model catalog integration."""

from typing import Dict, List, Optional
from pydantic import BaseModel


class NvidiaBuildModel(BaseModel):
    name: str
    publisher: str
    type: str
    capabilities: List[str]


def get_nvidia_build_catalog() -> List[NvidiaBuildModel]:
    """Returns official NVIDIA Build catalogue mappings."""
    return [
        NvidiaBuildModel(name="deepseek-v4-pro-0813", publisher="DeepSeek AI", type="LLM", capabilities=["coding", "reasoning", "agentic"]),
        NvidiaBuildModel(name="deepseek-v4-flash-0731", publisher="DeepSeek AI", type="LLM", capabilities=["coding", "reasoning", "agentic"]),
        NvidiaBuildModel(name="nemotron-3.5-lightning-30b-a3b", publisher="NVIDIA", type="LLM", capabilities=["text", "agents", "reasoning"]),
        NvidiaBuildModel(name="gemma-4-31b-it", publisher="Google", type="LLM", capabilities=["reasoning", "coding", "agentic"]),
    ]
