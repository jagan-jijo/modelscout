"""Tests for benchmark normalization, evidence levels, recency, and aggregation."""

import pytest
from modelscout.benchmarks.aggregation import aggregate_model_benchmarks
from modelscout.benchmarks.evidence import (
    EvidenceLevel,
    get_evidence_weight,
    validate_evidence_transfer,
)
from modelscout.benchmarks.normalization import normalize_benchmark_score
from modelscout.benchmarks.recency import compute_recency_weight
from modelscout.database.repository import DatabaseRepository
from modelscout.dataset.loader import load_dataset
from modelscout.models.metadata import ModelMetadata


def test_normalize_benchmark_scales():
    # Percentage benchmarks
    norm, m = normalize_benchmark_score("LiveBench", 88.5)
    assert norm == 88.5

    # ELO benchmarks: 1200 ELO maps to 0-100
    norm_elo, m_elo = normalize_benchmark_score("Chatbot Arena ELO", 1200.0)
    assert 40.0 <= norm_elo <= 60.0
    assert "elo" in m_elo


def test_evidence_weights_and_transfer_validation():
    assert get_evidence_weight(EvidenceLevel.DIRECT) == 1.0
    assert get_evidence_weight(EvidenceLevel.LINE_INTERPOLATED) < 0.9

    # Valid transfer: 70B to 65B within same family
    valid, reason = validate_evidence_transfer(70_000_000_000, 65_000_000_000, "Llama", "Llama")
    assert valid is True

    # Invalid transfer: 70B to 7B (ratio 10x) -> must be rejected!
    invalid, reason = validate_evidence_transfer(70_000_000_000, 7_000_000_000, "Llama", "Llama")
    assert invalid is False
    assert "disparity too large" in reason


def test_recency_weight_demotes_frozen_source():
    current_wt = compute_recency_weight("2026-08-20", tier="current", has_current_alternatives=True)
    frozen_wt = compute_recency_weight("2024-05-01", tier="frozen", has_current_alternatives=True)
    assert current_wt > frozen_wt


def test_benchmark_aggregation_with_repository():
    repo = DatabaseRepository(":memory:")
    catalog = load_dataset()
    repo.import_catalog(catalog)

    model = ModelMetadata(
        id="llama-3.3-70b-instruct",
        display_name="Llama-3.3-70B-Instruct",
        family="Llama-3.3",
        publisher="Meta",
        parameters=70_000_000_000,
        active_parameters=70_000_000_000,
    )
    result = aggregate_model_benchmarks(model, repo)
    assert result.has_direct_evidence is True
    assert result.confidence == "HIGH"
    assert result.composite_score >= 80.0
    assert len(result.evidence_records) >= 2
