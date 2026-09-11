"""Load the bundled split catalogue or a compatible single JSON file."""

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from modelscout.dataset.schema import (
    AppleSiliconRecord,
    BenchmarkResult,
    CanonicalModelRecord,
    CpuRecord,
    DatasetCatalog,
    GpuRecord,
    HuggingFaceModelRecord,
    NvidiaBuildModelRecord,
    OllamaModelRecord,
    SourceRecord,
)


def _extract_blocks_from_text(text: str) -> List[tuple[str, Any]]:
    """Extracts JSON blocks from a mixed text/markdown document."""
    pos = 0
    blocks = []
    while pos < len(text):
        m = re.search(r"(\[|\{)", text[pos:])
        if not m:
            break
        start = pos + m.start()
        char = m.group(1)
        matching_char = "]" if char == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == char:
                    depth += 1
                elif c == matching_char:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end != -1:
            chunk = text[start:end]
            try:
                parsed = json.loads(chunk)
                header = text[pos:start].strip()
                blocks.append((header, parsed))
                pos = end
            except Exception:
                pos = start + 1
        else:
            pos = start + 1
    return blocks


def parse_raw_dataset_blocks(blocks: List[tuple[str, Any]]) -> Dict[str, Any]:
    """Classifies extracted JSON blocks into catalog categories."""
    catalog_data: Dict[str, Any] = {
        "apple_silicon": [],
        "amd_cpus": [],
        "intel_cpus": [],
        "nvidia_gpus": [],
        "amd_gpus": [],
        "intel_gpus": [],
        "ollama_models": [],
        "nvidia_build_models": [],
        "huggingface_models": [],
        "canonical_models": [],
        "benchmarks": [],
        "sources": [],
    }

    for header, block in blocks:
        h_lower = header.lower()
        if "apple" in h_lower and isinstance(block, list):
            catalog_data["apple_silicon"] = block
        elif "amd cpu" in h_lower and isinstance(block, list):
            catalog_data["amd_cpus"] = block
        elif "intel cpu" in h_lower and isinstance(block, list):
            catalog_data["intel_cpus"] = block
        elif "nvidia gpu" in h_lower and isinstance(block, list):
            catalog_data["nvidia_gpus"] = block
        elif "amd gpu" in h_lower and isinstance(block, list):
            catalog_data["amd_gpus"] = block
        elif "intel gpu" in h_lower and isinstance(block, list):
            catalog_data["intel_gpus"] = block
        elif "ollama" in h_lower and isinstance(block, list):
            catalog_data["ollama_models"] = block
        elif "nvidia build" in h_lower and isinstance(block, list):
            catalog_data["nvidia_build_models"] = block
        elif "hugging face" in h_lower and isinstance(block, dict):
            catalog_data["huggingface_models"].append(block)
        elif "schema" in h_lower and isinstance(block, dict):
            catalog_data["canonical_models"].append(block)
        elif "sources" in h_lower and isinstance(block, list):
            catalog_data["sources"] = block

    return catalog_data


def load_dataset(file_path: Optional[str | Path] = None) -> DatasetCatalog:
    """Load an explicit catalogue, the checkout assets, or the installed copy."""
    override = file_path or os.environ.get("MODELSCOUT_DATASET")
    if override:
        supplied = Path(override)
        candidates = [supplied / "dataset.json" if supplied.is_dir() else supplied]
    else:
        candidates = [
            Path(__file__).resolve().parents[3] / "assets" / "dataset.json",
            Path(__file__).parent / "data" / "dataset.json",
        ]

    target = None
    for c in candidates:
        if c.is_file():
            target = c
            break

    if not target:
        raise FileNotFoundError(f"dataset.json not found in candidate paths: {[str(c) for c in candidates]}")

    content = target.read_text(encoding="utf-8")

    # Try 1: Standard structured JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "files" in data:
        merged: Dict[str, Any] = {
            "version": data.get("version", "1.0.0"),
            "updated_at": data.get("updated_at", ""),
        }
        for filename in data["files"]:
            part_path = target.parent / filename
            if not part_path.is_file():
                raise FileNotFoundError(f"Catalogue part is missing: {part_path}")
            part = json.loads(part_path.read_text(encoding="utf-8"))
            if not isinstance(part, dict):
                raise ValueError(f"Catalogue part must contain an object: {part_path}")
            duplicate = set(merged).intersection(part) - {"version", "updated_at"}
            if duplicate:
                raise ValueError(f"Duplicate catalogue sections in {part_path}: {sorted(duplicate)}")
            merged.update({k: v for k, v in part.items() if k not in {"version", "updated_at"}})
        return DatasetCatalog.model_validate(merged)

    if isinstance(data, dict) and any(
        k in data for k in ["apple_silicon", "amd_cpus", "canonical_models"]
    ):
        return DatasetCatalog.model_validate(data)

    # Try 2: Multi-block extraction
    blocks = _extract_blocks_from_text(content)
    if not blocks:
        raise ValueError(f"Could not parse valid JSON or multi-block structures from {target}")

    dict_data = parse_raw_dataset_blocks(blocks)
    return DatasetCatalog.model_validate(dict_data)
