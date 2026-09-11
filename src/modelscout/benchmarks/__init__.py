"""Benchmark package for evidence resolution, normalization, recency, and multi-source aggregation."""

from modelscout.benchmarks.aggregation import AggregatedBenchmarkResult, NormalizedEvidenceRecord, aggregate_model_benchmarks
from modelscout.benchmarks.engine import BenchmarkEngine
from modelscout.benchmarks.evidence import EvidenceLevel, get_evidence_weight, validate_evidence_transfer
from modelscout.benchmarks.normalization import normalize_benchmark_score
from modelscout.benchmarks.recency import compute_recency_weight

__all__ = [
    "BenchmarkEngine",
    "AggregatedBenchmarkResult",
    "NormalizedEvidenceRecord",
    "aggregate_model_benchmarks",
    "EvidenceLevel",
    "get_evidence_weight",
    "validate_evidence_transfer",
    "normalize_benchmark_score",
    "compute_recency_weight",
]
