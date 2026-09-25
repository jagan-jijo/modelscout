"""Tests for the runtime-model registry, popularity selection, and offline fallback."""

import json

import pytest

from modelscout.models import runtime_catalog
from modelscout.models.runtime_catalog import (
    RuntimeModel,
    build_runtime_report,
    load_runtime_models,
    load_runtime_registry,
    select_runtime_models,
)


def make_runtime_model(
    model_id: str,
    downloads: int,
    *,
    runtime: str = "airllm",
    last_modified: str = "2026-09-20T00:00:00+00:00",
) -> RuntimeModel:
    """Build a minimal record for selector tests."""
    return RuntimeModel(
        runtime=runtime,
        model_id=model_id,
        downloads=downloads,
        last_modified=last_modified,
    )


def fail_fetch(*args, **kwargs):
    """Simulate an unavailable metadata endpoint."""
    raise ConnectionError("offline")


def fake_fetch(model_id, **kwargs):
    """Return deterministic exact-ID metadata for enrichment tests."""
    return {
        "model_id": model_id,
        "downloads": 20_000,
        "likes": 42,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_modified": "2026-09-24T00:00:00+00:00",
        "metadata_source": "huggingface",
    }


enrichment_calls = []


def recording_fetch(model_id, **kwargs):
    """Record enrichment calls and return deterministic live metadata."""
    enrichment_calls.append((model_id, kwargs))
    return {
        "model_id": model_id,
        "downloads": 20_000,
        "likes": 42,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_modified": "2026-09-24T00:00:00+00:00",
        "metadata_source": "huggingface",
    }


def partial_fetch(model_id, **kwargs):
    """Return no live metadata for one record to exercise mixed sources."""
    if model_id == "Qwen/Qwen3.8-27B":
        return None
    return recording_fetch(model_id, **kwargs)


def null_optional_fetch(model_id, **kwargs):
    """Return a successful live lookup with no optional metadata values."""
    return {
        "model_id": model_id,
        "downloads": None,
        "likes": None,
        "created_at": None,
        "last_modified": None,
        "metadata_source": "huggingface",
    }


deadline_clock = {"now": 0.0}
deadline_calls = []


def deadline_fetch(model_id, **kwargs):
    """Advance the fake clock and return high live popularity metadata."""
    deadline_calls.append((model_id, kwargs))
    deadline_clock["now"] += 2.0
    return {
        "model_id": model_id,
        "source_downloads": 100_000,
        "metadata_source": "huggingface",
    }


def deadline_monotonic():
    """Return the fake monotonic clock used by deadline tests."""
    return deadline_clock["now"]


def empty_asset_candidates():
    """Return no default asset candidates for unavailable-catalogue tests."""
    return []


def test_bundled_registry_contains_approved_exact_ids():
    sections = load_runtime_models(enrich=False, limit=6)

    assert set(sections) == {"ollama", "airllm", "colibri"}
    assert {record.model_id for record in sections["airllm"]} == {
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-Flash-Next",
        "Qwen/Qwen3-235B-A22B",
        "deepseek-ai/DeepSeek-V3",
        "moonshotai/Kimi-K3",
    }
    assert {record.model_id for record in sections["ollama"]} == {
        "qwen3:8b",
        "qwen3:32b",
        "deepseek-r1:7b",
        "llama3.3:70b",
        "gemma3:12b",
        "mistral-small3.1",
    }

    colibri_by_id = {record.model_id: record for record in sections["colibri"]}
    assert colibri_by_id["Qwen/Qwen3.6-35B-A3B"].container_id == (
        "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64"
    )
    assert colibri_by_id["Justvugg/GLM-5.3-colibri-int4-g64"].container_id == (
        "Justvugg/GLM-5.3-colibri-int4-g64"
    )
    assert colibri_by_id["mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"].container_id == (
        "mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"
    )
    assert {
        record.model_id
        for record in sections["colibri"]
        if record.popularity_fallback
    } == {"mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"}


def test_bundled_registry_matches_reviewed_huggingface_snapshots():
    """Keep exact source IDs, counts, dates, and container corrections bundled."""
    sections = load_runtime_models(enrich=False, limit=6)
    expected_source_counts = {
        ("ollama", "Qwen/Qwen3-8B"): 12_343_793,
        ("ollama", "Qwen/Qwen3-32B"): 4_494_102,
        ("ollama", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"): 293_176,
        ("ollama", "meta-llama/Llama-3.3-70B-Instruct"): 916_904,
        ("ollama", "google/gemma-3-12b-it"): 588_625,
        ("ollama", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"): 498_907,
        ("airllm", "Qwen/Qwen3-32B"): 4_494_102,
        ("airllm", "Qwen/Qwen3.8-27B"): 6_765_008,
        ("airllm", "Qwen/Qwen3.8-Flash-Next"): 830_208,
        ("airllm", "Qwen/Qwen3-235B-A22B"): 370_481,
        ("airllm", "deepseek-ai/DeepSeek-V3"): 1_099_848,
        ("airllm", "moonshotai/Kimi-K3"): 1_774_987,
        ("colibri", "moonshotai/Kimi-K3"): 1_774_987,
        ("colibri", "deepseek-ai/DeepSeek-V4-Flash"): 1_403_245,
        ("colibri", "Qwen/Qwen3.8-Flash-Next"): 830_208,
        ("colibri", "Qwen/Qwen3.6-35B-A3B"): 3_138_204,
        ("colibri", "zai-org/GLM-5.3"): 1_131_785,
        ("colibri", "zai-org/GLM-5.2"): 908_535,
    }
    records = {
        (runtime, record.source_id): record
        for runtime, runtime_records in sections.items()
        for record in runtime_records
    }

    for key, expected_count in expected_source_counts.items():
        assert records[key].source_downloads == expected_count

    for record in sections["ollama"]:
        assert record.downloads == record.source_downloads
        assert "hardware-specific speed estimate" in record.notes

    glm_53 = records[("colibri", "zai-org/GLM-5.3")]
    glm_52 = records[("colibri", "zai-org/GLM-5.2")]
    qwen_36 = records[("colibri", "Qwen/Qwen3.6-35B-A3B")]
    assert glm_53.downloads == 14_523
    assert glm_53.container_downloads == 14_523
    assert glm_53.parameters == "744B"
    assert glm_53.active_parameters == "40B"
    assert glm_52.downloads == 5_717
    assert glm_52.container_downloads == 5_717
    assert glm_52.popularity_fallback is True
    assert qwen_36.downloads == 10_000
    assert qwen_36.source_downloads == 3_138_204
    assert records[("airllm", "Qwen/Qwen3-32B")].last_modified == (
        "2025-07-26T03:45:22.000Z"
    )
    assert glm_53.last_modified == "2026-09-04T06:41:23.000Z"
    assert glm_52.last_modified == "2026-09-01T11:36:01.000Z"


def test_report_contains_per_runtime_metadata_sources(monkeypatch):
    """Expose a source summary for each runtime and aggregate genuine mixtures."""
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", recording_fetch)

    report = build_runtime_report(limit=1, enrich=True, offline=False)

    assert report["metadata_sources"] == {
        "ollama": "bundled",
        "airllm": "huggingface",
        "colibri": "huggingface",
    }
    assert report["metadata_source"] == "mixed"
    assert report["runtimes"]["airllm"][0]["metadata_source"] == "huggingface"


def test_live_null_metadata_still_marks_records_as_huggingface(monkeypatch):
    """A successful live response is not downgraded when optional fields are null."""
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", null_optional_fetch)

    report = build_runtime_report(limit=1, enrich=True, offline=False)

    assert report["metadata_sources"]["airllm"] == "huggingface"
    assert report["metadata_sources"]["colibri"] == "huggingface"
    for runtime in ("airllm", "colibri"):
        assert all(row["metadata_source"] == "huggingface" for row in report["runtimes"][runtime])


def test_enrichment_preserves_container_downloads_legacy_meaning(monkeypatch):
    """Live source popularity must not overwrite a container's bundled count."""
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", recording_fetch)

    report = build_runtime_report(limit=6, enrich=True, offline=False)
    row = next(
        row
        for row in report["runtimes"]["colibri"]
        if row["model_id"] == "mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"
    )

    assert row["downloads"] == 5_717
    assert row["source_downloads"] == 20_000
    assert row["container_downloads"] == 5_717
    assert row["popularity_fallback"] is True


def test_enrichment_deadline_does_not_promote_unchecked_records(monkeypatch):
    """Deadline-limited enrichment leaves unchecked records on bundled metadata."""
    deadline_clock["now"] = 0.0
    deadline_calls.clear()
    monkeypatch.setattr(runtime_catalog.time, "monotonic", deadline_monotonic)
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", deadline_fetch)

    report = build_runtime_report(limit=6, enrich=True, offline=False)
    checked_ids = {model_id for model_id, _ in deadline_calls}

    assert deadline_calls
    assert len(deadline_calls) <= runtime_catalog.MAX_ENRICHMENT_LOOKUPS
    assert any("deadline" in warning.lower() for warning in report["warnings"])
    for runtime in ("airllm", "colibri"):
        for row in report["runtimes"][runtime]:
            expected_source = "huggingface" if row["source_id"] in checked_ids else "bundled"
            assert row["metadata_source"] == expected_source


def test_enrichment_lookup_cap_is_bounded(monkeypatch):
    """The configured lookup cap stops enrichment before exceeding the limit."""
    deadline_clock["now"] = 0.0
    deadline_calls.clear()
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", deadline_fetch)
    monkeypatch.setattr(runtime_catalog, "MAX_ENRICHMENT_LOOKUPS", 1)

    report = build_runtime_report(limit=6, enrich=True, offline=False)

    assert len(deadline_calls) == 1
    assert any("cap" in warning.lower() for warning in report["warnings"])


def test_runtime_report_rejects_unbounded_limit():
    """Keep direct callers within the same six-record surface as CLI/API."""
    with pytest.raises(ValueError):
        build_runtime_report(limit=7, enrich=False, offline=True)
    with pytest.raises(ValueError):
        select_runtime_models([], runtime="airllm", limit=7)


def test_select_runtime_models_prefers_ten_thousand_downloads():
    records = [
        make_runtime_model("Qwen/Qwen3-32B", 10_000),
        make_runtime_model("Qwen/Qwen3.8-27B", 9_999),
    ]

    selected = select_runtime_models(
        records,
        runtime="airllm",
        limit=1,
        primary=10_000,
        fallback=5_000,
    )

    assert [record.model_id for record in selected] == ["Qwen/Qwen3-32B"]
    assert selected[0].popularity_fallback is False


def test_select_runtime_models_uses_fallback_only_for_short_section():
    records = [
        make_runtime_model(
            "primary",
            12_000,
            runtime="colibri",
            last_modified="2026-09-19T00:00:00+00:00",
        ),
        make_runtime_model(
            "fallback",
            5_001,
            runtime="colibri",
            last_modified="2026-09-20T00:00:00+00:00",
        ),
    ]

    selected = select_runtime_models(
        records,
        runtime="colibri",
        limit=2,
        primary=10_000,
        fallback=5_000,
    )

    assert [record.model_id for record in selected] == ["primary", "fallback"]
    assert selected[0].popularity_fallback is False
    assert selected[1].popularity_fallback is True


def test_select_runtime_models_uses_deterministic_ties():
    records = [
        make_runtime_model("B/model", 10_000, last_modified="2026-09-20T00:00:00+00:00"),
        make_runtime_model("A/model", 10_000, last_modified="2026-09-20T00:00:00+00:00"),
    ]

    selected = select_runtime_models(records, runtime="airllm", limit=2)

    assert [record.model_id for record in selected] == ["A/model", "B/model"]


def test_runtime_record_serializes_exact_ids_and_metadata():
    record = RuntimeModel(
        runtime="colibri",
        model_id="Qwen/Qwen3.6-35B-A3B",
        container_id="Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64",
        downloads=10_500,
    )

    payload = record.to_dict()

    assert payload["model_id"] == "Qwen/Qwen3.6-35B-A3B"
    assert payload["container_id"] == "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64"
    assert payload["downloads"] == 10_500
    assert json.loads(json.dumps(payload)) == payload


def test_network_failure_keeps_bundled_snapshot(monkeypatch):
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", fail_fetch)

    report = build_runtime_report(limit=6, enrich=True, offline=False)

    assert set(report["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert report["metadata_source"] == "bundled"
    assert report["runtimes"]["airllm"][0]["metadata_source"] == "bundled"
    assert report["warnings"]


def test_exact_id_enrichment_updates_snapshot(monkeypatch):
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", fake_fetch)

    report = build_runtime_report(limit=6, enrich=True, offline=False)
    row = next(
        row
        for row in report["runtimes"]["airllm"]
        if row["model_id"] == "Qwen/Qwen3-32B"
    )

    assert row["downloads"] == 20_000
    assert row["likes"] == 42
    assert row["last_modified"] == "2026-09-24T00:00:00+00:00"
    assert row["metadata_source"] == "huggingface"
    assert report["metadata_source"] == "mixed"


def test_report_metadata_source_is_mixed_when_records_use_multiple_sources(monkeypatch):
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", partial_fetch)

    report = build_runtime_report(limit=6, enrich=True, offline=False)
    row_sources = {
        row["metadata_source"]
        for records in report["runtimes"].values()
        for row in records
    }

    assert report["metadata_source"] == "mixed"
    assert {"bundled", "huggingface"}.issubset(row_sources)


def test_colibri_source_and_container_popularity_are_independent(monkeypatch):
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", recording_fetch)

    offline_report = build_runtime_report(limit=6, enrich=False, offline=True)
    online_report = build_runtime_report(limit=6, enrich=True, offline=False)

    offline_row = next(
        row
        for row in offline_report["runtimes"]["colibri"]
        if row["model_id"]
        == "mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"
    )
    online_row = next(
        row
        for row in online_report["runtimes"]["colibri"]
        if row["model_id"]
        == "mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"
    )
    assert offline_row["source_downloads"] == 908_535
    assert offline_row["container_downloads"] == 5_717
    assert online_row["source_downloads"] == 20_000
    assert online_row["container_downloads"] == 5_717
    assert online_row["source_id"] == "zai-org/GLM-5.2"
    assert online_row["container_id"] == (
        "mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp"
    )


def test_limit_one_enriches_all_curated_records_under_cap(monkeypatch):
    enrichment_calls.clear()
    monkeypatch.setattr(runtime_catalog, "fetch_huggingface_model", recording_fetch)
    bundled = load_runtime_models(enrich=False, limit=6)
    expected_source_ids = {
        record.popularity_id
        for runtime in ("airllm", "colibri")
        for record in bundled[runtime]
    }

    report = build_runtime_report(limit=1, enrich=True, offline=False)

    assert set(report["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert expected_source_ids <= {model_id for model_id, _ in enrichment_calls}
    assert len(enrichment_calls) <= runtime_catalog.MAX_ENRICHMENT_LOOKUPS
    assert all(kwargs["timeout"] <= 1.0 for _, kwargs in enrichment_calls)


def test_missing_section_and_runtime_mismatch_warn_without_dropping_shape(tmp_path):
    registry = tmp_path / "runtime_models.json"
    registry.write_text(
        json.dumps(
            {
                "runtimes": {
                    "ollama": [],
                    "airllm": [
                        {
                            "runtime": "colibri",
                            "model_id": "wrong-section/model",
                            "downloads": 12000,
                        },
                        {
                            "model_id": "missing-required",
                            "downloads": 12000,
                        },
                    ],
                }
            }
        )
    )

    report = build_runtime_report(
        file_path=registry,
        enrich=False,
        offline=True,
    )

    assert set(report["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert any("missing" in warning.lower() and "colibri" in warning.lower() for warning in report["warnings"])
    assert any("mismatch" in warning.lower() for warning in report["warnings"])
    assert any("skipped invalid airllm" in warning.lower() for warning in report["warnings"])


def test_malformed_runtime_records_are_skipped_without_escaping(tmp_path):
    registry = tmp_path / "runtime_models.json"
    registry.write_text(
        json.dumps(
            {
                "runtimes": {
                    "ollama": [
                        {
                            "runtime": "ollama",
                            "model_id": "valid:tag",
                            "install_command": "ollama pull valid:tag",
                            "source_url": "https://ollama.com/library/valid",
                            "downloads": 1,
                        },
                        {
                            "runtime": "ollama",
                            "model_id": "bad:tag",
                            "install_command": "",
                            "source_url": "https://ollama.com/library/bad",
                            "downloads": "not-a-number",
                            "ram_gb": float("nan"),
                        },
                    ],
                    "airllm": [],
                    "colibri": [],
                }
            }
        )
    )

    records, warnings = load_runtime_registry(registry)

    assert [record.model_id for record in records["ollama"]] == ["valid:tag"]
    assert any("Skipped invalid ollama record" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("downloads", 1.5),
        ("source_downloads", 1.5),
        ("downloads", True),
        ("ram_gb", True),
        ("disk_gb", "nan"),
        ("popularity_fallback", "false"),
    ],
)
def test_runtime_model_rejects_fractional_boolean_and_nonfinite_values(field_name, value):
    """Reject invalid popularity and resource scalar values at the record boundary."""
    data = {
        "runtime": "airllm",
        "model_id": "bad/model",
        "install_command": "python -c 'load'",
        "source_url": "https://huggingface.co/bad/model",
        "downloads": 10,
        field_name: value,
    }

    with pytest.raises(ValueError):
        RuntimeModel.from_dict(data)


def test_runtime_model_requires_curated_install_and_source_identifiers():
    """Curated records cannot omit their installation or source URL metadata."""
    with pytest.raises(ValueError):
        RuntimeModel.from_dict(
            {
                "runtime": "airllm",
                "model_id": "missing/source",
                "downloads": 10,
            }
        )

    with pytest.raises(ValueError):
        RuntimeModel.from_dict(
            {
                "runtime": "colibri",
                "model_id": "missing/source",
                "install_command": "hf download missing/source",
                "downloads": 10,
            }
        )

    with pytest.raises(ValueError):
        RuntimeModel.from_dict(
            {
                "runtime": "colibri",
                "model_id": "container/model",
                "container_id": "container/model",
                "install_command": "hf download container/model",
                "source_url": "https://huggingface.co/container/model",
                "downloads": 10,
            }
        )


def test_runtime_model_rejects_nonfinite_and_negative_resource_values():
    with pytest.raises(ValueError):
        RuntimeModel.from_dict(
            {
                "runtime": "airllm",
                "model_id": "bad/model",
                "install_command": "python -c 'load'",
                "source_url": "https://huggingface.co/bad/model",
                "downloads": 10,
                "ram_gb": float("inf"),
            }
        )

    with pytest.raises(ValueError):
        RuntimeModel.from_dict(
            {
                "runtime": "colibri",
                "model_id": "bad/model",
                "install_command": "hf download bad/model",
                "source_url": "https://huggingface.co/bad/model",
                "downloads": 10,
                "disk_gb": -1,
            }
        )

    record = RuntimeModel(
        runtime="colibri",
        model_id="bad/model",
        install_command="hf download bad/model",
        source_url="https://huggingface.co/bad/model",
    )
    record.ram_gb = float("nan")
    with pytest.raises(ValueError):
        record.to_dict()


def test_missing_explicit_asset_uses_default_catalogue(tmp_path):
    missing = tmp_path / "missing-runtime-models.json"

    report = build_runtime_report(
        file_path=missing,
        enrich=False,
        offline=True,
    )

    assert len(report["runtimes"]["airllm"]) == 6
    assert any(
        "missing" in warning.lower() and "default" in warning.lower()
        for warning in report["warnings"]
    )


def test_invalid_explicit_asset_uses_default_catalogue(tmp_path):
    invalid = tmp_path / "invalid-runtime-models.json"
    invalid.write_text("{not-json")

    report = build_runtime_report(
        file_path=invalid,
        enrich=False,
        offline=True,
    )

    assert len(report["runtimes"]["colibri"]) == 6
    assert any(
        "invalid" in warning.lower() and "default" in warning.lower()
        for warning in report["warnings"]
    )


def test_unavailable_catalogue_reports_no_records(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_catalog, "_default_asset_candidates", empty_asset_candidates)
    missing = tmp_path / "missing-runtime-models.json"

    report = build_runtime_report(
        file_path=missing,
        enrich=False,
        offline=True,
    )

    assert set(report["runtimes"]) == {"ollama", "airllm", "colibri"}
    assert not any(report["runtimes"].values())
    assert any("no runtime records" in warning.lower() for warning in report["warnings"])
