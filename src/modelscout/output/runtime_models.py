"""Render curated runtime-model records for terminal and JSON surfaces."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from rich.box import ROUNDED
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from modelscout.output import _console

_RUNTIME_TITLES = {
    "ollama": "OLLAMA MODELS",
    "airllm": "AIRLLM COMPATIBLE MODELS",
    "colibri": "COLIBRI COMPATIBLE MODELS",
}


def _format_optional_number(value: Any, suffix: str = " GB") -> str:
    """Format an optional numeric catalogue value for a table cell."""
    if value is None:
        return "—"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _format_memory(record: Mapping[str, Any]) -> str:
    """Format reported VRAM and RAM without presenting them as estimates."""
    vram = record.get("reported_vram_gb", record.get("vram_gb"))
    ram = record.get("ram_gb")
    if vram is None and ram is None:
        return "—"
    vram_text = "CPU / —" if vram in (None, 0, 0.0) else _format_optional_number(vram)
    ram_text = _format_optional_number(ram)
    return f"VRAM {vram_text} / RAM {ram_text}"


def _format_updated(record: Mapping[str, Any]) -> str:
    """Return the source snapshot update timestamp."""
    return str(record.get("last_modified") or record.get("updated_at") or "—")


def _runtime_table(records: Sequence[Mapping[str, Any]]) -> Table:
    """Build a readable field/value table for one runtime section."""
    table = Table(
        box=ROUNDED,
        header_style="bold cyan",
        show_lines=True,
        expand=False,
    )
    table.add_column("Runtime model field", style="bold white", max_width=24)
    table.add_column("Value", overflow="fold", max_width=62)

    if not records:
        table.add_row("No qualifying records", "—")
        return table

    for record in records:
        model_id = str(record.get("model_id") or "—")
        source_id = str(record.get("source_id") or model_id)
        container_id = record.get("container_id")
        if model_id != source_id:
            table.add_row("Model ID", model_id)
        table.add_row("Source ID", source_id)
        if model_id != source_id:
            table.add_row("Registry model ID", model_id)
        if container_id:
            table.add_row("Container ID", str(container_id))
        table.add_row("Install", str(record.get("install_command") or "—"))
        table.add_row(
            "Source downloads",
            _format_optional_number(record.get("source_downloads"), suffix=""),
        )
        table.add_row(
            "Container downloads",
            _format_optional_number(record.get("container_downloads"), suffix=""),
        )
        table.add_row("Updated", _format_updated(record))
        table.add_row("VRAM/RAM", _format_memory(record))
        table.add_row("Disk", _format_optional_number(record.get("disk_gb")))
        table.add_row("Benchmark", str(record.get("benchmark") or "—"))
        table.add_row(
            "Source",
            str(record.get("source") or "—"),
        )
        table.add_row(
            "Source URL",
            str(record.get("source_url") or record.get("huggingface_url") or "—"),
        )
        table.add_row("Metadata source", str(record.get("metadata_source") or "—"))
        table.add_row("Notes", str(record.get("notes") or "—"))
        table.add_row("", "")
    return table


def _render_exact_identifiers(records: Sequence[Mapping[str, Any]]) -> None:
    """Print exact identifiers separately so narrow terminals cannot crop them."""
    for record in records:
        model_id = str(record.get("model_id") or "—")
        source_id = str(record.get("source_id") or model_id)
        container_id = record.get("container_id")
        if model_id != source_id:
            _console.console.print(
                f"Exact registry model ID: {model_id}",
                soft_wrap=True,
            )
        _console.console.print(f"Exact source ID: {source_id}", soft_wrap=True)
        if container_id:
            _console.console.print(
                f"Exact container ID: {container_id}",
                soft_wrap=True,
            )
        _console.console.print(
            f"Exact install command: {record.get('install_command', '—')}",
            soft_wrap=True,
        )


def _render_runtime_section(runtime: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Print one clearly titled runtime section."""
    title = _RUNTIME_TITLES.get(runtime, runtime.upper())
    _console.console.print(
        Panel(_runtime_table(records), title=title, box=ROUNDED),
        width=180,
    )
    _render_exact_identifiers(records)


def render_runtime_models_terminal(report: Mapping[str, Any]) -> None:
    """Render the grouped runtime report as three Rich sections."""
    runtimes = report.get("runtimes", {})
    if not isinstance(runtimes, Mapping):
        runtimes = {}
    for runtime in ("ollama", "airllm", "colibri"):
        records = runtimes.get(runtime, [])
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            records = []
        _render_runtime_section(
            runtime,
            [record for record in records if isinstance(record, Mapping)],
        )

    _console.console.print(
        f"Metadata source: {report.get('metadata_source', 'bundled')} | "
        f"Checked: {report.get('checked_at', '—')}"
    )
    warnings = report.get("warnings", [])
    if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
        for warning in warnings:
            _console.console.print(Text(f"Warning: {warning}", style="yellow"))


def render_runtime_models_json(report: Mapping[str, Any]) -> None:
    """Write the grouped runtime report as strict JSON on stdout."""
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def display_runtime_models_terminal(report: Mapping[str, Any]) -> None:
    """Compatibility alias for the runtime terminal renderer."""
    render_runtime_models_terminal(report)


def display_runtime_models_json(report: Mapping[str, Any]) -> None:
    """Compatibility alias for the runtime JSON renderer."""
    render_runtime_models_json(report)
