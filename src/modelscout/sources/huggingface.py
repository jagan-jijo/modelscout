"""Hugging Face API integration for discovering models, GGUFs, and metadata with offline fallback."""

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CACHE_DIR = Path.home() / ".cache" / "modelscout"
CACHE_FILE = CACHE_DIR / "hf_models_cache.json"
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours
HF_MODEL_CACHE_PREFIX = "model_detail_v1:"


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


def _read_cache_file() -> Dict[str, Any]:
    """Read the shared Hugging Face cache file when it is available."""
    if not CACHE_FILE.is_file():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        logger.exception("Could not read the Hugging Face cache: %s", error)
        return {}
    if not isinstance(data, dict):
        logger.error("The Hugging Face cache is not a JSON object")
        return {}
    return data


def _model_cache_key(model_id: str) -> str:
    """Return the versioned cache key for one exact model repository."""
    return f"{HF_MODEL_CACHE_PREFIX}{model_id}"


def _cache_entry_timestamp(entry: Dict[str, Any]) -> Optional[float]:
    """Return a valid per-entry cache timestamp, or None for legacy/invalid data."""
    value = entry.get("cached_at", entry.get("timestamp"))
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    return timestamp


def _read_model_cache(model_id: str, *, fresh_only: bool = False) -> Optional[Dict[str, Any]]:
    """Return cached metadata for an exact model, optionally requiring freshness."""
    cached = _read_cache_file()
    entry = cached.get(_model_cache_key(model_id))
    if not isinstance(entry, dict):
        return None
    has_entry_timestamp = "cached_at" in entry or "timestamp" in entry
    if has_entry_timestamp and _cache_entry_timestamp(entry) is None:
        return None
    if fresh_only:
        timestamp = _cache_entry_timestamp(entry)
        if timestamp is None or time.time() - timestamp >= CACHE_TTL_SECONDS:
            return None
    result = dict(entry)
    result.pop("cached_at", None)
    result["metadata_source"] = "cache"
    return result


def _save_model_cache(model_id: str, metadata: Dict[str, Any]) -> None:
    """Persist exact model metadata atomically without touching search freshness."""
    temporary_path: Optional[Path] = None
    try:
        cached = _read_cache_file()
        entry = dict(metadata)
        entry["cached_at"] = time.time()
        cached[_model_cache_key(model_id)] = entry
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=CACHE_FILE.parent,
            prefix=f".{CACHE_FILE.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(cached, indent=2))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, CACHE_FILE)
        temporary_path = None
    except (OSError, TypeError, ValueError, OverflowError) as error:
        logger.exception("Could not save the Hugging Face model cache: %s", error)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                logger.exception("Could not clean up the Hugging Face cache temp file: %s", error)


def _coerce_optional_int(value: Any) -> Optional[int]:
    """Convert an API or cache value to an optional integer."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric_value) or not numeric_value.is_integer() or numeric_value < 0:
        return None
    return int(numeric_value)


def _normalise_model_metadata(model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one exact Hugging Face API response to stable field names."""
    returned_id = payload.get("id") or payload.get("modelId")
    if returned_id and str(returned_id) != model_id:
        raise ValueError(
            f"Hugging Face returned {returned_id!r} for requested model {model_id!r}"
        )
    return {
        "model_id": model_id,
        "downloads": _coerce_optional_int(payload.get("downloads")),
        "likes": _coerce_optional_int(payload.get("likes")),
        "created_at": payload.get("createdAt") or payload.get("created_at"),
        "last_modified": payload.get("lastModified") or payload.get("last_modified"),
    }


def _model_detail_url(model_id: str) -> str:
    """Build the exact Hugging Face model-detail URL."""
    raw_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").strip()
    if not raw_endpoint.startswith(("http://", "https://")):
        raise ValueError("HF_ENDPOINT must start with http:// or https://")
    endpoint = raw_endpoint.rstrip("/")
    return f"{endpoint}/api/models/{quote(model_id, safe='/')}"


def fetch_huggingface_model(
    model_id: str,
    timeout: float = 6.0,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """Fetch popularity metadata for one exact model ID with cache fallback.

    The function only requests the small model-detail API payload. It never
    downloads model files. A stale exact-ID cache entry is returned when the
    network is unavailable so callers can still use their bundled snapshot.
    """
    if not model_id or not model_id.strip():
        return None
    model_id = model_id.strip()

    if use_cache:
        cached = _read_model_cache(model_id, fresh_only=True)
        if cached is not None:
            return cached

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(
                _model_detail_url(model_id),
                params={
                    "expand[]": [
                        "downloads",
                        "likes",
                        "createdAt",
                        "lastModified",
                    ]
                },
            )
        if response.status_code != 200:
            if use_cache:
                logger.warning(
                    "Hugging Face metadata request returned HTTP %s for %s",
                    response.status_code,
                    model_id,
                )
            return _read_model_cache(model_id) if use_cache else None
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Hugging Face model detail response is not an object")
        metadata = _normalise_model_metadata(model_id, payload)
        metadata["metadata_source"] = "huggingface"
        if use_cache:
            _save_model_cache(model_id, metadata)
        return metadata
    except Exception as error:
        logger.exception("Hugging Face metadata lookup failed for %s: %s", model_id, error)
        return _read_model_cache(model_id) if use_cache else None


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
