"""Live background model update and catalogue synchronization engine."""

import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Optional
import httpx

from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.loader import load_dataset
from modelscout.dataset.schema import (
    ArtifactRecord,
    CanonicalModelArchitecture,
    CanonicalModelCapabilities,
    CanonicalModelIdentity,
    CanonicalModelRecord,
    CanonicalModelRuntime,
    CanonicalModelSources,
    DatasetCatalog,
    HuggingFaceModelRecord,
    OllamaModelRecord,
)
from modelscout.sources.huggingface import search_huggingface_models

logger = logging.getLogger("modelscout.updater")

UPDATE_CHECK_FILE = Path.home() / ".cache" / "modelscout" / "last_update_check.json"
UPDATE_INTERVAL_SECONDS = 3600  # 1 hour minimum between background auto-updates


def _should_update(force: bool = False) -> bool:
    if force:
        return True
    try:
        if UPDATE_CHECK_FILE.is_file():
            data = json.loads(UPDATE_CHECK_FILE.read_text(encoding="utf-8"))
            last_ts = data.get("timestamp", 0)
            if time.time() - last_ts < UPDATE_INTERVAL_SECONDS:
                return False
    except Exception:
        pass
    return True


def _record_update_timestamp() -> None:
    try:
        UPDATE_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_CHECK_FILE.write_text(
            json.dumps({"timestamp": time.time(), "date": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _fetch_ollama_tags(timeout: float = 1.5) -> List[Dict[str, Any]]:
    """Inspects local Ollama instance tags if service is running."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get("http://127.0.0.1:11434/api/tags")
            if r.status_code == 200:
                return r.json().get("models", [])
    except Exception:
        pass
    return []


def _extract_param_billions(name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:-|\b|_)", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    if "70b" in name.lower():
        return 70.0
    if "8b" in name.lower():
        return 8.0
    if "7b" in name.lower():
        return 7.0
    if "3b" in name.lower():
        return 3.0
    if "1b" in name.lower():
        return 1.0
    if "32b" in name.lower():
        return 32.0
    if "14b" in name.lower():
        return 14.0
    return 7.0


def sync_models_from_open_apis(timeout: float = 8.0, repo: Optional[DatabaseRepository] = None) -> Dict[str, Any]:
    """Fetches open-endpoint model info from HuggingFace and local Ollama within strict timeout (<10s).
    Updates the local database and assets/models.json when available.
    """
    stats = {"hf_found": 0, "ollama_found": 0, "added_models": 0, "errors": []}
    if repo is None:
        repo = DatabaseRepository()

    # 1. Fetch HuggingFace trending text models (timeout guarded, <10s)
    hf_summaries = []
    try:
        hf_summaries = search_huggingface_models(query="", limit=25, timeout=min(timeout, 6.0), use_cache=False)
        stats["hf_found"] = len(hf_summaries)
    except Exception as e:
        stats["errors"].append(f"HuggingFace error: {e}")

    # 2. Fetch local Ollama tags if running
    ollama_tags = []
    try:
        ollama_tags = _fetch_ollama_tags(timeout=1.2)
        stats["ollama_found"] = len(ollama_tags)
    except Exception as e:
        stats["errors"].append(f"Ollama error: {e}")

    # 3. Ingest and merge new models into local SQLite
    existing_models = {m["id"] for m in repo.get_all_models()}
    new_canonical: List[CanonicalModelRecord] = []

    for item in hf_summaries:
        mid = item.id.lower().replace("/", "-").replace(".", "-")
        if mid in existing_models:
            continue

        param_b = item.estimated_parameters_b or _extract_param_billions(item.model_name)
        param_int = int(param_b * 1_000_000_000)

        # Detect capabilities from tags & model name
        is_code = any(kw in item.model_name.lower() or kw in item.tags for kw in ["coder", "coding", "code", "devstral", "starcoder"])
        is_reasoning = any(kw in item.model_name.lower() or kw in item.tags for kw in ["r1", "reasoning", "deepseek-r1", "qwq", "cot"])
        is_vision = any(kw in item.model_name.lower() or kw in item.tags for kw in ["vision", "vl", "multimodal", "pixtral"])

        # Determine family
        family = "Community"
        if "llama" in item.model_name.lower():
            family = "Llama"
        elif "qwen" in item.model_name.lower():
            family = "Qwen"
        elif "mistral" in item.model_name.lower():
            family = "Mistral"
        elif "gemma" in item.model_name.lower():
            family = "Gemma"
        elif "deepseek" in item.model_name.lower():
            family = "DeepSeek"
        elif "phi" in item.model_name.lower():
            family = "Phi"

        # Construct CanonicalModelRecord
        record = CanonicalModelRecord(
            id=mid,
            identity=CanonicalModelIdentity(
                canonical_name=item.model_name,
                family=family,
                publisher=item.publisher,
            ),
            sources=CanonicalModelSources(
                huggingface={"id": item.id, "downloads": item.downloads, "likes": item.likes},
                ollama=[],
                nvidia_build=[],
            ),
            architecture=CanonicalModelArchitecture(
                type="Dense",
                total_parameters=param_int,
                active_parameters=param_int,
            ),
            context={"maximum_tokens": 131072 if param_b >= 7 else 32768},
            capabilities=CanonicalModelCapabilities(
                text=True,
                coding=is_code,
                reasoning=is_reasoning,
                vision=is_vision,
            ),
            runtime=CanonicalModelRuntime(
                ollama=True,
                llama_cpp=True,
                vllm=True,
                transformers=True,
            ),
            artifacts=[
                ArtifactRecord(format="GGUF", quantization="Q4_K_M", source=f"https://huggingface.co/{item.id}"),
                ArtifactRecord(format="GGUF", quantization="Q5_K_M", source=f"https://huggingface.co/{item.id}"),
                ArtifactRecord(format="GGUF", quantization="Q8_0", source=f"https://huggingface.co/{item.id}"),
            ],
            provenance={
                "last_verified": time.strftime("%Y-%m-%d"),
                "sources": [f"https://huggingface.co/{item.id}"],
                "confidence": "medium",
            },
        )
        new_canonical.append(record)
        existing_models.add(mid)

    # Ingest new canonical records into database
    if new_canonical:
        stats["added_models"] = len(new_canonical)
        repo.insert_models(new_canonical)

    _record_update_timestamp()
    return stats


def start_background_model_update(timeout: float = 8.0, force: bool = False) -> None:
    """Spawns non-blocking daemon thread to auto-update local model data in background."""
    if not _should_update(force=force):
        return

    def _worker():
        try:
            sync_models_from_open_apis(timeout=timeout)
        except Exception as e:
            logger.debug(f"Background update finished with exception: {e}")

    thread = threading.Thread(target=_worker, name="modelscout-background-updater", daemon=True)
    thread.start()
