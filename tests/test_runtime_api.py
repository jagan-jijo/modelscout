"""Tests for the read-only runtime-model API endpoint."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from modelscout.web.app import app

client = TestClient(app, base_url="http://127.0.0.1")


def test_api_runtime_models_returns_same_grouped_shape():
    response = client.get("/api/runtime-models?limit=6&offline=true")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert payload["metadata_source"] == "bundled"
    assert payload["metadata_sources"] == {
        "ollama": "bundled",
        "airllm": "bundled",
        "colibri": "bundled",
    }
    assert isinstance(payload["warnings"], list)
    assert "Qwen/Qwen3-32B" in {
        row["model_id"] for row in payload["runtimes"]["airllm"]
    }
    assert "moonshotai/Kimi-K3" in {
        row["model_id"] for row in payload["runtimes"]["colibri"]
    }


def test_api_runtime_models_skips_malformed_records_without_500(tmp_path, monkeypatch):
    """Return an honest empty report when a custom asset contains invalid records."""
    registry = tmp_path / "runtime_models.json"
    registry.write_text(
        json.dumps(
            {
                "runtimes": {
                    "ollama": [
                        {
                            "runtime": "ollama",
                            "model_id": "bad:tag",
                            "downloads": 1.5,
                            "ram_gb": True,
                        }
                    ],
                    "airllm": [],
                    "colibri": [],
                }
            }
        )
    )
    monkeypatch.setenv("MODELSCOUT_RUNTIME_MODELS", str(registry))

    response = client.get("/api/runtime-models?offline=true")

    assert response.status_code == 200
    payload = response.json()
    assert not any(payload["runtimes"].values())
    assert any("skipped invalid" in warning.lower() for warning in payload["warnings"])


def test_api_runtime_models_validates_limit():
    response = client.get("/api/runtime-models?limit=0&offline=true")

    assert response.status_code == 422


def test_api_runtime_models_does_not_expose_filesystem_paths(tmp_path, monkeypatch):
    missing_registry = tmp_path / "missing-runtime-models.json"
    monkeypatch.setenv("MODELSCOUT_RUNTIME_MODELS", str(missing_registry))

    response = client.get("/api/runtime-models?offline=true")

    assert response.status_code == 200
    payload = response.json()
    warnings_text = json.dumps(payload["warnings"])
    assert set(payload["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert "/Volumes" not in warnings_text
    assert str(missing_registry) not in warnings_text
    assert str(Path.home()) not in warnings_text
