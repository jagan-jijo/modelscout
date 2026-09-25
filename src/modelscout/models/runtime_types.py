"""Typed records for models documented as compatible with local runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _first_value(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present value from a record mapping."""
    for key in keys:
        if key in data:
            return data[key]
    return default


def _string_value(data: Mapping[str, Any], *keys: str, default: str = "") -> str:
    """Return a non-null string value from a record mapping."""
    value = _first_value(data, *keys)
    if value is None:
        return default
    return str(value)


def _optional_string(data: Mapping[str, Any], *keys: str) -> str | None:
    """Return an optional non-empty string value from a record mapping."""
    value = _first_value(data, *keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_nonnegative_int(value: Any, field_name: str) -> int:
    """Convert a popularity value to a finite, non-negative integer."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if isinstance(value, int):
        number = value
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{field_name} must be a non-negative integer") from error
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field_name} must be a non-negative integer")
        number = int(numeric_value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _coerce_optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    """Convert an optional popularity value, rejecting malformed numbers."""
    if value is None or value == "":
        return None
    return _coerce_nonnegative_int(value, field_name)


def _coerce_nonnegative_float(value: Any, field_name: str) -> float:
    """Convert a resource value to a finite, non-negative number."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a finite non-negative number") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return number


def _coerce_optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    """Convert an optional resource value, rejecting malformed numbers."""
    if value is None or value == "":
        return None
    return _coerce_nonnegative_float(value, field_name)


def _normalise_runtime_values(record: Any) -> None:
    """Normalize and validate numeric runtime fields before serialization."""
    record.downloads = _coerce_nonnegative_int(record.downloads, "downloads")
    record.source_downloads = _coerce_optional_nonnegative_int(
        record.source_downloads,
        "source_downloads",
    )
    record.container_downloads = _coerce_optional_nonnegative_int(
        record.container_downloads,
        "container_downloads",
    )
    record.likes = _coerce_nonnegative_int(record.likes, "likes")
    if not isinstance(record.popularity_fallback, bool):
        raise ValueError("popularity_fallback must be a boolean")
    record.reported_vram_gb = _coerce_optional_nonnegative_float(
        record.reported_vram_gb,
        "reported_vram_gb",
    )
    record.ram_gb = _coerce_optional_nonnegative_float(record.ram_gb, "ram_gb")
    record.disk_gb = _coerce_optional_nonnegative_float(record.disk_gb, "disk_gb")


@dataclass
class RuntimeModel:
    """A curated runtime compatibility record with installation metadata."""

    runtime: str
    model_id: str
    container_id: str | None = None
    source_id: str | None = None
    display_name: str = ""
    family: str = ""
    architecture: str = ""
    parameters: str | None = None
    active_parameters: str | None = None
    install_command: str = ""
    source_url: str = ""
    huggingface_url: str = ""
    source: str = ""
    downloads: int = 0
    source_downloads: int | None = None
    container_downloads: int | None = None
    likes: int = 0
    created_at: str | None = None
    last_modified: str | None = None
    reported_vram_gb: float | None = None
    ram_gb: float | None = None
    disk_gb: float | None = None
    benchmark: Any = None
    notes: str = ""
    metadata_source: str = "bundled"
    popularity_fallback: bool = False

    def __post_init__(self) -> None:
        """Keep URL and source-ID aliases consistent for direct construction."""
        if not self.source_url:
            self.source_url = self.huggingface_url
        if not self.huggingface_url and self.runtime in {"airllm", "colibri"}:
            self.huggingface_url = self.source_url
        if not self.source_id:
            self.source_id = self.model_id
        if self.source_downloads is None:
            self.source_downloads = self.downloads
        if self.container_id == self.model_id and self.container_downloads is None:
            self.container_downloads = self.downloads
        _normalise_runtime_values(self)

    @property
    def popularity_id(self) -> str:
        """Return the source checkpoint ID used for popularity filtering."""
        return self.source_id or self.model_id

    @property
    def source_popularity(self) -> int:
        """Return source-checkpoint downloads used by popularity selection."""
        return self.source_downloads if self.source_downloads is not None else self.downloads

    @property
    def display(self) -> str:
        """Return the human-readable model name."""
        return self.display_name or self.model_id

    @property
    def vram_gb(self) -> float | None:
        """Return the source-reported VRAM value."""
        return self.reported_vram_gb

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeModel":
        """Build a runtime record from bundled or cached JSON data."""
        model_id = _string_value(data, "model_id", "id").strip()
        if not model_id:
            raise ValueError("Runtime model records require a model_id")

        source_id = _optional_string(data, "source_id", "popularity_id")
        runtime = _string_value(data, "runtime", default="unknown").strip().lower()
        source_url = _string_value(
            data, "source_url", "huggingface_url", "model_url", "url"
        ).strip()
        huggingface_url = _string_value(data, "huggingface_url", "model_url").strip()
        if not huggingface_url and runtime in {"airllm", "colibri"}:
            huggingface_url = source_url
        downloads_value = _first_value(data, "downloads")
        if "downloads" in data and downloads_value is None:
            raise ValueError("downloads must not be null")
        downloads = _coerce_optional_nonnegative_int(
            downloads_value, "downloads"
        ) or 0
        likes_value = _first_value(data, "likes")
        if "likes" in data and likes_value is None:
            raise ValueError("likes must not be null")
        source_downloads = _coerce_optional_nonnegative_int(
            _first_value(data, "source_downloads"),
            "source_downloads",
        )
        install_command = _string_value(
            data, "install_command", "install", "command"
        ).strip()
        source = _string_value(data, "source", "documentation_url").strip()
        if runtime in {"ollama", "airllm", "colibri"}:
            if not install_command:
                raise ValueError("Curated runtime records require an install command")
            if not (source_url or huggingface_url or source):
                raise ValueError("Curated runtime records require a source URL")
        container_id = _optional_string(data, "container_id", "container")
        if runtime in {"airllm", "colibri"} and container_id and not source_id:
            raise ValueError("Container runtime records require a source ID")
        popularity_fallback = _first_value(
            data, "popularity_fallback", default=False
        )
        if not isinstance(popularity_fallback, bool):
            raise ValueError("popularity_fallback must be a boolean")
        return cls(
            runtime=runtime,
            model_id=model_id,
            container_id=container_id,
            source_id=source_id or model_id,
            display_name=_string_value(
                data, "display_name", "display", default=model_id
            ),
            family=_string_value(data, "family"),
            architecture=_string_value(data, "architecture", "architecture_type"),
            parameters=_optional_string(data, "parameters", "parameter_count"),
            active_parameters=_optional_string(
                data, "active_parameters", "active_parameter_count"
            ),
            install_command=install_command,
            source_url=source_url,
            huggingface_url=huggingface_url,
            source=source,
            downloads=downloads,
            source_downloads=source_downloads if source_downloads is not None else downloads,
            container_downloads=_coerce_optional_nonnegative_int(
                _first_value(data, "container_downloads"),
                "container_downloads",
            ),
            likes=_coerce_optional_nonnegative_int(likes_value, "likes") or 0,
            created_at=_optional_string(data, "created_at", "createdAt"),
            last_modified=_optional_string(
                data, "last_modified", "lastModified", "updated_at"
            ),
            reported_vram_gb=_coerce_optional_nonnegative_float(
                _first_value(data, "reported_vram_gb", "reported_vram", "vram_gb"),
                "reported_vram_gb",
            ),
            ram_gb=_coerce_optional_nonnegative_float(
                _first_value(data, "ram_gb", "ram"),
                "ram_gb",
            ),
            disk_gb=_coerce_optional_nonnegative_float(
                _first_value(data, "disk_gb", "disk"),
                "disk_gb",
            ),
            benchmark=_first_value(data, "benchmark", "benchmark_notes"),
            notes=_string_value(data, "notes"),
            metadata_source=_string_value(
                data, "metadata_source", default="bundled"
            ),
            popularity_fallback=popularity_fallback,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record without changing exact runtime identifiers."""
        _normalise_runtime_values(self)
        return {
            "runtime": self.runtime,
            "model_id": self.model_id,
            "container_id": self.container_id,
            "source_id": self.source_id,
            "display_name": self.display_name,
            "family": self.family,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "active_parameters": self.active_parameters,
            "install_command": self.install_command,
            "source_url": self.source_url,
            "huggingface_url": self.huggingface_url,
            "source": self.source,
            "downloads": self.downloads,
            "source_downloads": self.source_downloads,
            "container_downloads": self.container_downloads,
            "likes": self.likes,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "reported_vram_gb": self.reported_vram_gb,
            "ram_gb": self.ram_gb,
            "disk_gb": self.disk_gb,
            "benchmark": self.benchmark,
            "notes": self.notes,
            "metadata_source": self.metadata_source,
            "popularity_fallback": self.popularity_fallback,
        }

    def model_dump(self) -> dict[str, Any]:
        """Provide a Pydantic-style serialization alias for callers."""
        return self.to_dict()
