"""Canonical model metadata representation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from modelscout.models.artifacts import ModelArtifact


class ModelMetadata(BaseModel):
    id: str
    display_name: str
    family: str
    publisher: str
    parameters: int
    active_parameters: int
    architecture: str = "Dense"  # Dense, MoE
    context_length: int = 131072
    is_moe: bool = False
    num_experts: Optional[int] = None
    active_experts: Optional[int] = None
    artifacts: List[ModelArtifact] = Field(default_factory=list)
    capabilities: Dict[str, bool] = Field(
        default_factory=lambda: {
            "text": True,
            "vision": False,
            "audio": False,
            "video": False,
            "reasoning": False,
            "coding": False,
            "tool_calling": False,
            "agents": False,
        }
    )
    ollama_name: Optional[str] = None
    nvidia_build_name: Optional[str] = None
    huggingface_id: Optional[str] = None
    license: Optional[str] = None
    created_at: Optional[str] = None
