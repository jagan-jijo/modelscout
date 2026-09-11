"""Ollama integration: query local installed models and verify model tag availability."""

from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel


class LocalOllamaModel(BaseModel):
    name: str
    size_bytes: int
    modified_at: str
    digest: str


def get_local_ollama_models(timeout: float = 1.0) -> List[LocalOllamaModel]:
    """Queries running local Ollama instance for pulled models."""
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            models = []
            for item in data.get("models", []):
                models.append(
                    LocalOllamaModel(
                        name=item.get("name", ""),
                        size_bytes=item.get("size", 0),
                        modified_at=item.get("modified_at", ""),
                        digest=item.get("digest", ""),
                    )
                )
            return models
    except Exception:
        pass
    return []
