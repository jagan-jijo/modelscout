"""Model discovery combining local database, Ollama catalog, and live Hugging Face cache."""

from typing import List, Optional
from modelscout.database.repository import DatabaseRepository
from modelscout.models.artifacts import ModelArtifact
from modelscout.models.metadata import ModelMetadata
from modelscout.models.normalization import parse_moe_parameters, parse_parameters_str


def discover_available_models(repo: Optional[DatabaseRepository] = None) -> List[ModelMetadata]:
    """Discovers all candidate models from database and catalogues."""
    if repo is None:
        repo = DatabaseRepository()

    db_models = repo.get_all_models()
    candidates: List[ModelMetadata] = []

    for m in db_models:
        total_p = m["total_parameters"]
        active_p = m["active_parameters"] or total_p
        is_moe = m["architecture_type"] == "MoE" or active_p < total_p

        arts = []
        for a in m.get("artifacts", []):
            gb = round(a["file_size_bytes"] / (1024**3), 2) if a.get("file_size_bytes") else None
            if not gb:
                art_est = ModelArtifact.estimate(total_p, a["quantization"])
                arts.append(art_est)
            else:
                arts.append(
                    ModelArtifact(
                        format=a["format"],
                        quantization=a["quantization"],
                        file_size_bytes=a["file_size_bytes"],
                        file_size_gb=gb,
                        source=a.get("source"),
                    )
                )

        if not arts:
            # Default artifacts
            arts = [
                ModelArtifact.estimate(total_p, "Q4_K_M"),
                ModelArtifact.estimate(total_p, "Q5_K_M"),
                ModelArtifact.estimate(total_p, "Q8_0"),
            ]

        meta = ModelMetadata(
            id=m["id"],
            display_name=m["canonical_name"],
            family=m["family_id"].replace("-", " ").title(),
            publisher=m["publisher"],
            parameters=total_p,
            active_parameters=active_p,
            architecture=m["architecture_type"],
            context_length=m["context_length"],
            is_moe=is_moe,
            artifacts=arts,
            capabilities={
                "text": bool(m["has_text"]),
                "vision": bool(m["has_vision"]),
                "audio": bool(m["has_audio"]),
                "video": False,
                "reasoning": bool(m["has_reasoning"]),
                "coding": bool(m["has_coding"]),
                "tool_calling": bool(m["has_tool_calling"]),
                "agents": bool(m["has_agents"]),
            },
            ollama_name=m["ollama_name"],
            nvidia_build_name=m["nvidia_build_name"],
            huggingface_id=m["huggingface_id"],
        )
        candidates.append(meta)

    return candidates
