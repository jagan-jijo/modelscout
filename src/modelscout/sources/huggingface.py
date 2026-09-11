"""Hugging Face API integration for discovering models, GGUFs, and metadata with offline fallback."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

CACHE_DIR = Path.home() / ".cache" / "modelscout"
CACHE_FILE = CACHE_DIR / "hf_models_cache.json"
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours


class HuggingFaceModelSummary(BaseModel):
    id: str
    model_name: str
    publisher: str = "Unknown"
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = "text-generation"
    tags: List[str] = Field(default_factory=list)
    has_gguf: bool = False
    estimated_parameters_b: Optional[float] = None
    created_at: Optional[str] = None


def _get_cache() -> Optional[Dict[str, Any]]:
    try:
        if CACHE_FILE.is_file():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            cache_time = data.get("timestamp", 0)
            if time.time() - cache_time < CACHE_TTL_SECONDS:
                return data
    except Exception:
        pass
    return None


def _save_cache(cache_key: str, data: List[Dict[str, Any]]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = {}
        if CACHE_FILE.is_file():
            try:
                cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                cached = {}
        cached["timestamp"] = time.time()
        cached[cache_key] = data
        CACHE_FILE.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_parameters_from_name(name: str) -> Optional[float]:
    """Extracts approximate parameter count from model name (e.g., '7b', '70B', '0.5B')."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:-|\b|_)", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def search_huggingface_models(
    query: str = "",
    limit: int = 40,
    sort: str = "downloads",
    direction: int = -1,
    timeout: float = 6.0,
    use_cache: bool = True,
) -> List[HuggingFaceModelSummary]:
    """Queries Hugging Face API for text-generation/GGUF models with strict timeout and cache fallback."""
    cache_key = f"search_{query}_{sort}_{limit}"
    if use_cache:
        cached = _get_cache()
        if cached and cache_key in cached:
            try:
                return [HuggingFaceModelSummary.model_validate(item) for item in cached[cache_key]]
            except Exception:
                pass

    url = "https://huggingface.co/api/models"
    params = {
        "pipeline_tag": "text-generation",
        "sort": sort,
        "direction": str(direction),
        "limit": str(limit),
    }
    if query:
        params["search"] = query

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, params=params)
            if r.status_code == 200:
                results: List[HuggingFaceModelSummary] = []
                for item in r.json():
                    mid = item.get("id", "")
                    if not mid:
                        continue
                    parts = mid.split("/", 1)
                    publisher = parts[0] if len(parts) > 1 else "Community"
                    mname = parts[-1]
                    tags = item.get("tags", [])
                    has_gguf = "gguf" in tags or "gguf" in mid.lower()
                    params_b = _extract_parameters_from_name(mname)

                    summary = HuggingFaceModelSummary(
                        id=mid,
                        model_name=mname,
                        publisher=publisher,
                        downloads=item.get("downloads", 0),
                        likes=item.get("likes", 0),
                        pipeline_tag=item.get("pipeline_tag", "text-generation"),
                        tags=tags,
                        has_gguf=has_gguf,
                        estimated_parameters_b=params_b,
                        created_at=item.get("createdAt"),
                    )
                    results.append(summary)

                if results and use_cache:
                    _save_cache(cache_key, [m.model_dump() for m in results])
                return results
    except Exception:
        # Graceful fallback: Return cached data if available even if expired
        try:
            if CACHE_FILE.is_file():
                cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if cache_key in cached:
                    return [HuggingFaceModelSummary.model_validate(item) for item in cached[cache_key]]
        except Exception:
            pass

    return []


def fetch_huggingface_ggufs(
    limit: int = 30,
    timeout: float = 6.0,
) -> List[HuggingFaceModelSummary]:
    """Specifically fetches top trending GGUF repositories from Hugging Face."""
    return search_huggingface_models(query="gguf", limit=limit, sort="downloads", timeout=timeout)
