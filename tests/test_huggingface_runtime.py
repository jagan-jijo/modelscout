"""Tests for exact-ID Hugging Face metadata lookup and cache fallback."""

import json
import time

import httpx

from modelscout.sources import huggingface


class FakeResponse:
    """Successful exact-model API response used by cache tests."""

    status_code = 200

    def json(self):
        """Return a minimal model-detail payload."""
        return {
            "id": "Qwen/Qwen3-32B",
            "downloads": 12345,
            "likes": 67,
            "createdAt": "2026-01-01T00:00:00+00:00",
            "lastModified": "2026-09-24T00:00:00+00:00",
        }


class FakeClient:
    """HTTP client double that records exact-ID requests."""

    calls = []

    def __init__(self, **kwargs):
        """Accept the production client options without opening a socket."""
        self.kwargs = kwargs

    def __enter__(self):
        """Enter the fake client context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Leave the fake client context."""
        return False

    def get(self, url, params=None):
        """Record the request and return the fixture response."""
        self.calls.append((url, params))
        return FakeResponse()


class PerModelResponse:
    """Successful response for an exact model selected by URL."""

    status_code = 200

    def __init__(self, model_id):
        """Store the model ID used to synthesize metadata."""
        self.model_id = model_id

    def json(self):
        """Return distinct metadata for the requested model."""
        return {
            "id": self.model_id,
            "downloads": 22222,
            "likes": 22,
            "createdAt": "2026-01-01T00:00:00+00:00",
            "lastModified": "2026-09-24T00:00:00+00:00",
        }


class PerModelClient(FakeClient):
    """HTTP client double that returns model-specific responses."""

    calls = []

    def get(self, url, params=None):
        """Record the request and synthesize a response from its exact ID."""
        model_id = url.split("/api/models/", 1)[1]
        self.calls.append(model_id)
        return PerModelResponse(model_id)


class FailingClient(FakeClient):
    """HTTP client double for exercising stale-cache fallback."""

    def get(self, url, params=None):
        """Raise the same bounded network error as a failed lookup."""
        raise httpx.ConnectError("offline")


def test_live_metadata_does_not_truncate_fractional_or_boolean_counts():
    """Normalize malformed live counts without fabricating integer values."""
    metadata = huggingface._normalise_model_metadata(
        "bad/model",
        {
            "id": "bad/model",
            "downloads": 1.5,
            "likes": True,
        },
    )

    assert metadata["downloads"] is None
    assert metadata["likes"] is None


def test_exact_model_lookup_writes_versioned_cache(tmp_path, monkeypatch):
    FakeClient.calls = []
    cache_file = tmp_path / "hf_models_cache.json"
    cache_file.write_text(json.dumps({"timestamp": 123, "search_legacy": []}))
    monkeypatch.setattr(huggingface.httpx, "Client", FakeClient)
    monkeypatch.setattr(huggingface, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(huggingface, "CACHE_FILE", cache_file)

    metadata = huggingface.fetch_huggingface_model("Qwen/Qwen3-32B")

    assert metadata is not None
    assert metadata["downloads"] == 12345
    assert metadata["metadata_source"] == "huggingface"
    assert FakeClient.calls[0][0].endswith("/api/models/Qwen/Qwen3-32B")
    assert FakeClient.calls[0][1]["expand[]"] == [
        "downloads",
        "likes",
        "createdAt",
        "lastModified",
    ]
    cache = json.loads(cache_file.read_text())
    assert cache["timestamp"] == 123
    assert "search_legacy" in cache
    assert "model_detail_v1:Qwen/Qwen3-32B" in cache
    assert cache["model_detail_v1:Qwen/Qwen3-32B"]["cached_at"] > 0


def test_exact_model_lookup_returns_stale_cache_on_network_error(tmp_path, monkeypatch):
    cache_file = tmp_path / "hf_models_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "timestamp": 0,
                "model_detail_v1:Qwen/Qwen3-32B": {
                    "model_id": "Qwen/Qwen3-32B",
                    "downloads": 10000,
                    "likes": 1,
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "last_modified": "2025-01-02T00:00:00+00:00",
                },
            }
        )
    )
    monkeypatch.setattr(huggingface.httpx, "Client", FailingClient)
    monkeypatch.setattr(huggingface, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(huggingface, "CACHE_FILE", cache_file)

    metadata = huggingface.fetch_huggingface_model("Qwen/Qwen3-32B")

    assert metadata is not None
    assert metadata["downloads"] == 10000
    assert metadata["metadata_source"] == "cache"


def test_exact_model_cache_freshness_is_per_entry(tmp_path, monkeypatch):
    now = time.time()
    fresh_id = "Qwen/Qwen3-32B"
    stale_id = "Qwen/Qwen3.8-27B"
    cache_file = tmp_path / "hf_models_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "timestamp": now,
                f"model_detail_v1:{fresh_id}": {
                    "model_id": fresh_id,
                    "downloads": 11111,
                    "likes": 11,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "last_modified": "2026-01-02T00:00:00+00:00",
                    "cached_at": now,
                },
                f"model_detail_v1:{stale_id}": {
                    "model_id": stale_id,
                    "downloads": 7777,
                    "likes": 7,
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "last_modified": "2025-01-02T00:00:00+00:00",
                    "cached_at": 0,
                },
            }
        )
    )
    PerModelClient.calls = []
    monkeypatch.setattr(huggingface.httpx, "Client", PerModelClient)
    monkeypatch.setattr(huggingface, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(huggingface, "CACHE_FILE", cache_file)

    fresh = huggingface.fetch_huggingface_model(fresh_id)
    stale = huggingface.fetch_huggingface_model(stale_id)

    assert fresh is not None
    assert fresh["downloads"] == 11111
    assert stale is not None
    assert stale["downloads"] == 22222
    assert PerModelClient.calls == [stale_id]
