"""Regression coverage for the existing fixed-hardware recommendation path."""

from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.loader import load_dataset
from modelscout.hardware.detector import detect_system_hardware
from modelscout.models.runtime_catalog import build_runtime_report
from modelscout.recommendation.ranking import generate_recommendations

BASELINE_CATALOG = load_dataset()
EXPECTED_RECOMMENDATION_IDS = [
    "deepseek-r1-14b",
    "phi-4-14b",
    "qwen3-8b",
    "deepseek-r1-7b",
    "qwen2.5-coder-7b-instruct",
    "qwen3-14b",
    "deepseek-r1-8b",
    "gemma-4-e2b",
    "qwen2.5-coder-14b-instruct",
    "codestral-22b",
    "qwen3-4b",
    "mistral-small-24b-instruct-2501",
    "gpt-oss-20b",
    "devstral-small-24b",
    "openai-gpt-oss-20b",
]


def test_runtime_catalog_does_not_change_fixed_hardware_recommendation_ids():
    repo = DatabaseRepository(":memory:")
    repo.import_catalog(BASELINE_CATALOG)
    hardware = detect_system_hardware(
        cpu_override="AMD Ryzen 9 7950X",
        gpu_override="RTX 4090",
        ram_override_gb=32,
        vram_override_gb=16,
    )
    runtime_report = build_runtime_report(limit=1, enrich=False, offline=True)
    assert set(runtime_report["runtimes"]) == {"ollama", "airllm", "colibri"}

    report = generate_recommendations(
        hardware,
        profile="general",
        top_n=15,
        repo=repo,
    )

    actual_ids = [recommendation.model_id for recommendation in report.recommendations]
    canonical_ids = {model.id for model in BASELINE_CATALOG.canonical_models}
    assert set(EXPECTED_RECOMMENDATION_IDS).issubset(canonical_ids)
    assert actual_ids == EXPECTED_RECOMMENDATION_IDS
    assert len(actual_ids) == 15
