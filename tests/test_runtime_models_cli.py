"""Tests for the grouped runtime-model Typer command."""

import json
from io import StringIO

from rich.console import Console
from typer.testing import CliRunner

from modelscout.cli.main import app
from modelscout.output import _console
from modelscout.output.runtime_models import render_runtime_models_terminal

runner = CliRunner()


def test_runtime_cli_json_is_strict_and_grouped():
    result = runner.invoke(app, ["runtimes", "--offline", "--json", "--limit", "6"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert payload["checked_at"]
    assert isinstance(payload["metadata_source"], str)
    assert payload["metadata_sources"] == {
        "ollama": "bundled",
        "airllm": "bundled",
        "colibri": "bundled",
    }
    assert isinstance(payload["warnings"], list)
    assert "Qwen/Qwen3-32B" in {
        row["model_id"] for row in payload["runtimes"]["airllm"]
    }
    assert "qwen3:8b" in {
        row["model_id"] for row in payload["runtimes"]["ollama"]
    }
    assert "moonshotai/Kimi-K3" in {
        row["model_id"] for row in payload["runtimes"]["colibri"]
    }


def test_runtime_cli_renders_three_sections_and_exact_ids():
    result = runner.invoke(app, ["runtimes", "--offline", "--limit", "6"])

    assert result.exit_code == 0, result.output
    assert "OLLAMA MODELS" in result.output
    assert "AIRLLM COMPATIBLE MODELS" in result.output
    assert "COLIBRI COMPATIBLE MODELS" in result.output
    assert "Qwen/Qwen3-32B" in result.output
    assert "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64" in result.output
    assert "Source ID" in result.output
    assert "Source downloads" in result.output
    assert "Container downloads" in result.output
    assert "1,774,987" in result.output
    assert "Source URL" in result.output
    assert "Source" in result.output
    assert "github.com/lyogavin/airllm" in result.output
    assert "Benchmark" in result.output
    assert "Notes" in result.output
    assert "Model/source ID" not in result.output
    assert "HF downloads" not in result.output
    assert "ollama pull qwen3:8b" in result.output


def test_runtime_cli_rejects_limit_outside_one_to_six():
    result = runner.invoke(app, ["runtimes", "--offline", "--limit", "0"])

    assert result.exit_code != 0


def test_runtime_output_star_imports_keep_legacy_exports():
    cli_exports = {}
    json_exports = {}
    exec("from modelscout.cli.output import *", cli_exports)
    exec("from modelscout.output.json_output import *", json_exports)

    assert {"Any", "SystemHardware", "render_scan_json", "render_runtime_models_json"}.issubset(
        cli_exports
    )
    assert {
        "json",
        "effective_quant_type",
        "HardwareInfo",
        "ModelInfo",
        "display_json",
        "display_upgrade_json",
        "render_runtime_models_json",
    }.issubset(json_exports)


def test_runtime_renderer_prints_bracketed_warnings_as_text(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(_console, "console", Console(file=output, width=120))

    render_runtime_models_terminal(
        {
            "runtimes": {"ollama": [], "airllm": [], "colibri": []},
            "warnings": ["[danger] bracketed warning"],
            "metadata_source": "bundled",
            "checked_at": "2026-09-24T00:00:00+00:00",
        }
    )

    assert "[danger] bracketed warning" in output.getvalue()


def test_runtime_renderer_uses_explicit_ids_downloads_and_source_row(monkeypatch):
    """Render source documentation and exact IDs without ambiguous duplicate lines."""
    output = StringIO()
    monkeypatch.setattr(_console, "console", Console(file=output, width=180))

    render_runtime_models_terminal(
        {
            "runtimes": {
                "ollama": [],
                "airllm": [],
                "colibri": [
                    {
                        "model_id": "container/model",
                        "source_id": "source/model",
                        "container_id": "container/model",
                        "source": "https://framework.example/docs",
                        "source_downloads": 11,
                        "container_downloads": 22,
                        "downloads": 22,
                        "benchmark": "CPU-only",
                        "ram_gb": 16,
                        "disk_gb": 32,
                        "notes": "Use the hardware estimate.",
                    }
                ],
            },
            "warnings": [],
            "metadata_source": "bundled",
            "checked_at": "2026-09-24T00:00:00+00:00",
        }
    )

    rendered = output.getvalue()
    assert "https://framework.example/docs" in rendered
    assert "Source downloads" in rendered
    assert "Container downloads" in rendered
    assert "source/model" in rendered
    assert "container/model" in rendered
    assert rendered.count("Exact source ID: source/model") == 1
    assert "Model/source ID" not in rendered
