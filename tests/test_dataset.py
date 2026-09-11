"""Tests for dataset loading, schema validation, and database importing."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.importer import import_dataset_to_db
from modelscout.dataset.loader import load_dataset
from modelscout.dataset.schema import BenchmarkResult, DatasetCatalog, GpuRecord
from modelscout.dataset.validator import validate_catalog


def test_load_and_validate_canonical_dataset():
    catalog = load_dataset()
    assert len(catalog.apple_silicon) >= 15
    assert len(catalog.amd_cpus) >= 19
    assert len(catalog.intel_cpus) >= 15
    assert len(catalog.nvidia_gpus) >= 26
    assert len(catalog.canonical_models) >= 25
    assert len(catalog.benchmarks) >= 10
    assert len(catalog.sources) >= 7
    val_res = validate_catalog(catalog)
    assert val_res.is_valid is True
    assert len(val_res.errors) == 0


def test_catalogue_is_split_by_subject():
    assets = Path(__file__).parents[1] / "assets"
    manifest = json.loads((assets / "dataset.json").read_text())
    assert manifest["files"] == ["cpus.json", "gpus.json", "models.json"]
    assert "amd_cpus" in json.loads((assets / "cpus.json").read_text())
    assert "nvidia_gpus" in json.loads((assets / "gpus.json").read_text())
    assert "canonical_models" in json.loads((assets / "models.json").read_text())

def test_dataset_loads_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_dataset().canonical_models) >= 10
    assert DatabaseRepository(tmp_path / "fresh.db").get_stats()["models"] >= 10


def test_missing_explicit_dataset_does_not_silently_use_defaults(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing)
    monkeypatch.setenv("MODELSCOUT_DATASET", str(missing))
    with pytest.raises(FileNotFoundError):
        load_dataset()


def test_invalid_structured_data_does_not_become_empty_catalogue(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"canonical_models": [{"id": "incomplete"}]}')
    with pytest.raises(ValidationError):
        load_dataset(invalid)


def test_validator_catches_invalid_records():
    # Construct invalid catalog
    bad_catalog = DatasetCatalog(
        nvidia_gpus=[
            GpuRecord(vendor="NVIDIA", name="Fake GPU", vram_gb=10000.0),  # Impossible VRAM
        ],
        benchmarks=[
            BenchmarkResult(model_id="test", benchmark="Fake", score=-5.0, source="Test"),  # Impossible score
        ],
    )
    val_res = validate_catalog(bad_catalog)
    assert val_res.is_valid is False
    assert len(val_res.errors) >= 2


def test_idempotent_sqlite_import():
    val_res, stats1 = import_dataset_to_db(None, ":memory:")
    assert val_res.is_valid is True
    assert stats1["models"] > 0
    assert stats1["gpus"] > 0
    repo = DatabaseRepository(":memory:")
    catalog = load_dataset()
    repo.import_catalog(catalog)
    first = repo.get_stats()
    repo.import_catalog(catalog)
    assert repo.get_stats() == first


def test_repository_autocomplete_search():
    repo = DatabaseRepository(":memory:")
    catalog = load_dataset()
    repo.import_catalog(catalog)

    # Search GPU by alias
    gpus_4090 = repo.search_gpus("4090")
    assert len(gpus_4090) >= 1
    assert any("4090" in g["name"] for g in gpus_4090)

    # Search CPU by family
    cpus_ryzen = repo.search_processors("Ryzen")
    assert len(cpus_ryzen) >= 1
    assert any("Ryzen" in c["name"] for c in cpus_ryzen)

    # Multi-vendor coverage tests: Apple, Intel, AMD
    cpus_apple = repo.search_processors("apple")
    assert len(cpus_apple) >= 10
    assert all(c["vendor"] == "Apple" for c in cpus_apple)
    assert any("M3" in c["name"] for c in cpus_apple)

    cpus_intel = repo.search_processors("intel")
    assert len(cpus_intel) >= 20
    assert all(c["vendor"] == "Intel" for c in cpus_intel)

    cpus_amd = repo.search_processors("amd")
    assert len(cpus_amd) >= 20
    assert all(c["vendor"] == "AMD" for c in cpus_amd)

    # Verify clock frequency metadata is returned
    assert any(c.get("frequency_ghz") for c in cpus_intel)
    assert any(c.get("frequency_ghz") for c in cpus_amd)
    assert any(c.get("frequency_ghz") for c in cpus_apple)

    # Unfiltered catalog search returns all vendors
    all_cpus = repo.search_processors("")
    all_vendors = {c["vendor"] for c in all_cpus}
    assert {"Apple", "Intel", "AMD"}.issubset(all_vendors)
