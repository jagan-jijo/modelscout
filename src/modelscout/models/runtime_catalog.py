"""Load, enrich, select, and serialize the curated runtime-model catalogue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from functools import cmp_to_key
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from modelscout.models.runtime_types import RuntimeModel
from modelscout.sources.huggingface import fetch_huggingface_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

RUNTIME_NAMES = ("ollama", "airllm", "colibri")
MAX_RUNTIME_LIMIT = 6
PRIMARY_DOWNLOAD_THRESHOLD = 10_000
FALLBACK_DOWNLOAD_THRESHOLD = 5_000
RUNTIME_ASSET_ENV = "MODELSCOUT_RUNTIME_MODELS"
ENRICHMENT_TOTAL_TIMEOUT_SECONDS = 3.0
ENRICHMENT_REQUEST_TIMEOUT_SECONDS = 1.0
MAX_ENRICHMENT_LOOKUPS = 12


def _default_asset_candidates() -> list[Path]:
    """Return checkout and installed-package asset candidates."""
    module_path = Path(__file__).resolve()
    return [
        module_path.parents[3] / "assets" / "runtime_models.json",
        module_path.parents[1] / "dataset" / "data" / "runtime_models.json",
        module_path.parent / "data" / "runtime_models.json",
    ]


def _explicit_asset_candidate(file_path: str | Path | None) -> Path | None:
    """Return the explicitly requested asset path, if one was configured."""
    override = file_path or os.environ.get(RUNTIME_ASSET_ENV)
    if not override:
        return None
    supplied = Path(override)
    return supplied / "runtime_models.json" if supplied.is_dir() else supplied


def _asset_candidates(file_path: str | Path | None) -> list[Path]:
    """Return explicit candidates or the default asset candidates."""
    explicit_candidate = _explicit_asset_candidate(file_path)
    if explicit_candidate is not None:
        return [explicit_candidate]
    return _default_asset_candidates()


def _find_asset(candidates: Sequence[Path]) -> Path | None:
    """Return the first existing asset from a candidate sequence."""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_runtime_asset(file_path: str | Path | None = None) -> Path:
    """Resolve an explicit asset, falling back to the default when possible."""
    candidates = _asset_candidates(file_path)
    target = _find_asset(candidates)
    if target is not None:
        return target
    if _explicit_asset_candidate(file_path) is not None:
        target = _find_asset(_default_asset_candidates())
        if target is not None:
            return target
    raise FileNotFoundError("No runtime catalogue is available.")


def _read_runtime_payload(target: Path) -> Mapping[str, Any]:
    """Read and validate one runtime asset file."""
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"Could not read runtime catalogue: {target}") from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Runtime catalogue is not valid JSON: {target}") from error

    if not isinstance(payload, Mapping):
        raise ValueError("Runtime catalogue must contain a JSON object")
    runtime_payload = payload.get("runtimes", payload)
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("Runtime catalogue 'runtimes' value must be an object")
    return runtime_payload


def load_runtime_registry(
    file_path: str | Path | None = None,
) -> tuple[dict[str, list[RuntimeModel]], list[str]]:
    """Load curated runtime records with explicit-asset fallback warnings."""
    explicit_candidate = _explicit_asset_candidate(file_path)
    target = resolve_runtime_asset(file_path)
    warnings: list[str] = []
    if explicit_candidate is not None and not explicit_candidate.is_file():
        warnings.append(
            "Configured runtime catalogue was missing; using the default catalogue."
        )
    try:
        runtime_payload = _read_runtime_payload(target)
    except (FileNotFoundError, ValueError) as error:
        default_target = _find_asset(_default_asset_candidates())
        if (
            explicit_candidate is None
            or default_target is None
            or default_target == target
        ):
            raise error
        runtime_payload = _read_runtime_payload(default_target)
        warnings.append(
            "Configured runtime catalogue was invalid; using the default catalogue."
        )

    records: dict[str, list[RuntimeModel]] = {
        runtime: [] for runtime in RUNTIME_NAMES
    }
    for runtime in RUNTIME_NAMES:
        if runtime not in runtime_payload:
            message = f"Runtime section {runtime!r} is missing; using an empty section."
            warnings.append(message)
            logger.warning(message)
            raw_records = []
        else:
            raw_records = runtime_payload[runtime]
        if not isinstance(raw_records, list):
            message = f"Runtime section {runtime!r} is not a list; using an empty section."
            warnings.append(message)
            logger.warning(message)
            continue
        if not raw_records and runtime in runtime_payload:
            message = f"Runtime section {runtime!r} is empty; no records are available."
            warnings.append(message)
            logger.warning(message)
            continue
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                message = f"Skipped invalid {runtime} record at index {index}."
                warnings.append(message)
                logger.warning(message)
                continue
            declared_runtime = raw_record.get("runtime")
            declared_runtime_name = (
                str(declared_runtime).strip().lower() if declared_runtime else ""
            )
            if declared_runtime_name and declared_runtime_name != runtime:
                message = (
                    f"Dropped {runtime} record at index {index}: runtime mismatch "
                    f"({declared_runtime!r})."
                )
                warnings.append(message)
                logger.warning(message)
                continue
            try:
                record_data = dict(raw_record)
                if not declared_runtime_name:
                    record_data["runtime"] = runtime
                record = RuntimeModel.from_dict(record_data)
                if not declared_runtime_name or record.runtime != runtime:
                    record = replace(record, runtime=runtime)
                records[runtime].append(record)
            except (TypeError, ValueError, OverflowError) as error:
                message = f"Skipped invalid {runtime} record at index {index}: {error}."
                warnings.append(message)
                logger.exception(message)
    return records, warnings


def _compare_runtime_models(left: RuntimeModel, right: RuntimeModel) -> int:
    """Compare records using the catalogue's stable descending sort order."""
    left_modified = left.last_modified or ""
    right_modified = right.last_modified or ""
    if left_modified != right_modified:
        return -1 if left_modified > right_modified else 1
    if left.source_popularity != right.source_popularity:
        return -1 if left.source_popularity > right.source_popularity else 1
    if left.model_id != right.model_id:
        return -1 if left.model_id < right.model_id else 1
    return 0


def sort_runtime_models(records: Sequence[RuntimeModel]) -> list[RuntimeModel]:
    """Return runtime records sorted by freshness, popularity, and exact ID."""
    return sorted(records, key=cmp_to_key(_compare_runtime_models))


def select_runtime_models(
    records: Sequence[RuntimeModel],
    *,
    runtime: str,
    limit: int,
    primary: int = PRIMARY_DOWNLOAD_THRESHOLD,
    fallback: int = FALLBACK_DOWNLOAD_THRESHOLD,
) -> list[RuntimeModel]:
    """Select primary records first, admitting fallback records only to fill a short section."""
    if limit < 0 or limit > MAX_RUNTIME_LIMIT:
        raise ValueError(f"Runtime model limit must be between 0 and {MAX_RUNTIME_LIMIT}")
    if primary < 0 or fallback < 0 or fallback > primary:
        raise ValueError("Popularity thresholds must satisfy 0 <= fallback <= primary")

    runtime_records = sort_runtime_models(
        [record for record in records if record.runtime == runtime]
    )
    primary_records = [
        record
        for record in runtime_records
        if not record.popularity_fallback and record.source_popularity >= primary
    ]
    fallback_records = [
        record
        for record in runtime_records
        if record.popularity_fallback
        or fallback <= record.source_popularity < primary
    ]
    if len(primary_records) >= limit:
        selected = primary_records[:limit]
    else:
        selected = primary_records + fallback_records[: limit - len(primary_records)]
    return [
        replace(
            record,
            popularity_fallback=(
                record.popularity_fallback or record.source_popularity < primary
            ),
        )
        for record in selected
    ]


def _merge_runtime_metadata(
    record: RuntimeModel,
    metadata: Mapping[str, Any],
) -> RuntimeModel:
    """Merge source metadata while leaving bundled container popularity intact."""
    updates: dict[str, Any] = {}
    source_downloads = metadata.get("source_downloads")
    if source_downloads is None:
        source_downloads = metadata.get("downloads")
    if source_downloads is not None:
        updates["source_downloads"] = source_downloads
        if not record.container_id:
            updates["downloads"] = source_downloads
    for field_name in ("likes", "created_at", "last_modified"):
        value = metadata.get(field_name)
        if value is not None:
            updates[field_name] = value
    metadata_source = metadata.get("metadata_source")
    updates["metadata_source"] = (
        str(metadata_source) if metadata_source else "huggingface"
    )
    return replace(record, **updates)


def _enrichment_candidates(
    records: Sequence[RuntimeModel],
    limit: int,
) -> list[RuntimeModel]:
    """Return all curated records; the final limit is applied after enrichment."""
    return sort_runtime_models(
        [record for record in records if record.runtime in {"airllm", "colibri"}]
    )


def _enrich_runtime_records(
    records: Sequence[RuntimeModel],
    *,
    limit: int,
) -> tuple[list[RuntimeModel], list[str]]:
    """Enrich a bounded candidate set without fetching containers or weights."""
    enriched_by_id: dict[str, Mapping[str, Any] | None] = {}
    warnings: list[str] = []
    deadline = time.monotonic() + ENRICHMENT_TOTAL_TIMEOUT_SECONDS
    for record in _enrichment_candidates(records, limit):
        popularity_id = record.popularity_id
        if popularity_id in enriched_by_id:
            continue
        if len(enriched_by_id) >= MAX_ENRICHMENT_LOOKUPS:
            message = "The runtime metadata lookup cap was reached; using bundled snapshots."
            warnings.append(message)
            logger.warning(message)
            break
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            message = "The runtime metadata deadline was reached; using bundled snapshots."
            warnings.append(message)
            logger.warning(message)
            break
        request_timeout = min(
            ENRICHMENT_REQUEST_TIMEOUT_SECONDS,
            remaining_seconds,
        )
        try:
            metadata = fetch_huggingface_model(
                popularity_id,
                timeout=request_timeout,
                use_cache=True,
            )
        except Exception as error:
            message = f"Could not enrich {popularity_id!r}: {error}."
            warnings.append(message)
            logger.exception(message)
            metadata = None
        if metadata is None:
            message = (
                f"No live metadata was available for {popularity_id!r}; "
                "using the bundled snapshot."
            )
            warnings.append(message)
        enriched_by_id[popularity_id] = metadata

    enriched: list[RuntimeModel] = []
    for record in records:
        if record.runtime not in {"airllm", "colibri"}:
            enriched.append(record)
            continue
        metadata = enriched_by_id.get(record.popularity_id)
        if metadata is None:
            enriched.append(record)
            continue
        try:
            enriched.append(_merge_runtime_metadata(record, metadata))
        except (TypeError, ValueError, OverflowError) as error:
            message = (
                f"Could not apply live metadata for {record.popularity_id!r}: "
                f"{error}."
            )
            warnings.append(message)
            logger.exception(message)
            enriched.append(record)
    return enriched, warnings


def _select_runtime_sections(
    records: Mapping[str, Sequence[RuntimeModel]],
    *,
    limit: int,
) -> dict[str, list[RuntimeModel]]:
    """Select each runtime section without changing Ollama compatibility."""
    selected: dict[str, list[RuntimeModel]] = {}
    for runtime in RUNTIME_NAMES:
        section = list(records.get(runtime, []))
        if runtime == "ollama":
            selected[runtime] = [
                replace(record, popularity_fallback=False)
                for record in sort_runtime_models(section)[:limit]
            ]
        else:
            selected[runtime] = select_runtime_models(
                section,
                runtime=runtime,
                limit=limit,
            )
    return selected


def _checked_at() -> str:
    """Return the current UTC metadata check timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _public_warning(message: str) -> str:
    """Remove filesystem details from warnings returned to API consumers."""
    lowered = message.lower()
    if "runtime catalogue unavailable" in lowered or "candidate paths" in lowered:
        return "Runtime catalogue unavailable; no runtime records were loaded."
    if "could not enrich" in lowered or "could not apply live metadata" in lowered:
        return "Runtime metadata enrichment failed; using the bundled snapshot."
    path_markers = ("/Volumes/", "/Users/", "/private/", "/var/folders/", "/tmp/", ":\\")
    if any(marker in message for marker in path_markers):
        return "Runtime catalogue warning; no records were loaded for that request."
    return message


def _load_and_select(
    *,
    enrich: bool,
    limit: int,
    offline: bool,
    file_path: str | Path | None,
) -> tuple[dict[str, list[RuntimeModel]], list[str]]:
    """Load, optionally enrich, and select runtime sections."""
    if limit < 0 or limit > MAX_RUNTIME_LIMIT:
        raise ValueError(f"Runtime model limit must be between 0 and {MAX_RUNTIME_LIMIT}")
    try:
        records, warnings = load_runtime_registry(file_path)
    except (FileNotFoundError, TypeError, ValueError, OverflowError) as error:
        logger.exception("Runtime catalogue unavailable: %s", error)
        message = "Runtime catalogue unavailable; no runtime records were loaded."
        return (
            {runtime: [] for runtime in RUNTIME_NAMES},
            [message],
        )

    all_records = [
        record
        for runtime in RUNTIME_NAMES
        for record in records.get(runtime, [])
    ]
    if enrich and not offline:
        all_records, enrichment_warnings = _enrich_runtime_records(
            all_records,
            limit=limit,
        )
        warnings.extend(enrichment_warnings)
        records = {
            runtime: [
                record for record in all_records if record.runtime == runtime
            ]
            for runtime in RUNTIME_NAMES
        }

    return _select_runtime_sections(records, limit=limit), warnings


def load_runtime_models(
    *,
    enrich: bool = True,
    limit: int = 6,
    offline: bool = False,
    file_path: str | Path | None = None,
) -> dict[str, list[RuntimeModel]]:
    """Load selected runtime records from the bundled catalogue.

    ``enrich`` performs exact Hugging Face metadata lookups. Set ``offline``
    to force the bundled snapshot and avoid all network access.
    """
    selected, _warnings = _load_and_select(
        enrich=enrich,
        limit=limit,
        offline=offline,
        file_path=file_path,
    )
    return selected


def _runtime_metadata_source(records: Sequence[RuntimeModel]) -> str:
    """Summarize metadata sources for one runtime section."""
    if not records:
        return "unavailable"
    sources = {record.metadata_source for record in records if record.metadata_source}
    if len(sources) > 1:
        return "mixed"
    if "huggingface" in sources:
        return "huggingface"
    if "cache" in sources:
        return "cache"
    return "bundled"


def _report_metadata_sources(
    report: Mapping[str, Sequence[RuntimeModel]],
) -> dict[str, str]:
    """Return a metadata source summary for every runtime section."""
    return {
        runtime: _runtime_metadata_source(report.get(runtime, []))
        for runtime in RUNTIME_NAMES
    }


def _report_metadata_source(
    report: Mapping[str, Sequence[RuntimeModel]],
    metadata_sources: Mapping[str, str] | None = None,
) -> str:
    """Summarize metadata sources while ignoring genuinely empty runtimes."""
    sources = metadata_sources or _report_metadata_sources(report)
    populated_sources = {
        source for source in sources.values() if source != "unavailable"
    }
    if len(populated_sources) > 1:
        return "mixed"
    if "mixed" in populated_sources:
        return "mixed"
    if "huggingface" in populated_sources:
        return "huggingface"
    if "cache" in populated_sources:
        return "cache"
    if "bundled" in populated_sources:
        return "bundled"
    return "unavailable"


def build_runtime_report(
    *,
    limit: int = 6,
    enrich: bool = True,
    offline: bool = False,
    file_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the strict grouped payload shared by the CLI and API."""
    selected, warnings = _load_and_select(
        enrich=enrich,
        limit=limit,
        offline=offline,
        file_path=file_path,
    )
    if not any(selected.values()):
        warnings.append("No qualifying runtime models were found.")
    metadata_sources = _report_metadata_sources(selected)
    payload = {
        "runtimes": {
            runtime: [record.to_dict() for record in selected[runtime]]
            for runtime in RUNTIME_NAMES
        },
        "checked_at": _checked_at(),
        "metadata_source": _report_metadata_source(selected, metadata_sources),
        "metadata_sources": metadata_sources,
        "warnings": [_public_warning(warning) for warning in warnings],
    }
    return payload


def load_runtime_catalog(
    *,
    enrich: bool = True,
    limit: int = 6,
    offline: bool = False,
    file_path: str | Path | None = None,
) -> dict[str, list[RuntimeModel]]:
    """Compatibility alias for callers that name the registry a catalogue."""
    return load_runtime_models(
        enrich=enrich,
        limit=limit,
        offline=offline,
        file_path=file_path,
    )
