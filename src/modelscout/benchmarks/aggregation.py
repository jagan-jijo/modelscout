"""Multi-source benchmark aggregation with evidence grading, recency weighting, and provenance."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from modelscout.benchmarks.evidence import (
    EvidenceLevel,
    get_evidence_weight,
    validate_evidence_transfer,
)
from modelscout.benchmarks.normalization import normalize_benchmark_score
from modelscout.benchmarks.recency import compute_recency_weight
from modelscout.database.repository import DatabaseRepository
from modelscout.models.metadata import ModelMetadata


class NormalizedEvidenceRecord(BaseModel):
    benchmark_name: str
    original_score: float
    normalized_score: float
    source: str
    source_url: Optional[str] = None
    date: Optional[str] = None
    evidence_type: str = "direct"
    tier: str = "current"
    weight_applied: float = 1.0


class AggregatedBenchmarkResult(BaseModel):
    composite_score: float
    primary_benchmark: str
    primary_source: str
    evidence_type: str
    confidence: str  # HIGH, MEDIUM, LOW
    has_direct_evidence: bool
    evidence_records: List[NormalizedEvidenceRecord] = Field(default_factory=list)


def aggregate_model_benchmarks(
    model: ModelMetadata,
    repo: DatabaseRepository,
) -> AggregatedBenchmarkResult:
    """Aggregates multi-source benchmark evidence for candidate model."""
    raw_records = repo.get_model_benchmarks(model.id)

    # If no records for exact ID, check canonical name or family variants
    evidence_type = "direct"
    has_direct = len(raw_records) > 0

    if not raw_records:
        # Check if there is a verified base/instruct variant
        clean_id = model.id.replace("-instruct", "")
        raw_records = repo.get_model_benchmarks(clean_id)
        if raw_records:
            evidence_type = "variant"
        else:
            # Fallback to family line interpolation if parameter size is comparable
            fam_records = []
            conn = repo.get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """SELECT b.*, m.total_parameters FROM benchmarks b
                       JOIN models m ON m.id = b.model_id
                       WHERE m.family_id = ?""",
                    (model.family.lower().replace(" ", "-"),),
                )
                for row in cur.fetchall():
                    r_dict = dict(row)
                    source_params = r_dict.get("total_parameters", model.parameters)
                    is_valid, _ = validate_evidence_transfer(
                        source_params=source_params,
                        target_params=model.parameters,
                        source_family=model.family,
                        target_family=model.family,
                    )
                    if is_valid:
                        fam_records.append(r_dict)
            finally:
                if repo.db_path != ":memory:":
                    conn.close()

            if fam_records:
                raw_records = fam_records
                evidence_type = "line_interpolated"

    if not raw_records:
        # Fallback default score estimate based on parameter scaling when no benchmarks exist
        # e.g., 7B ~ 72, 14B ~ 78, 32B ~ 83, 70B ~ 86
        p_billions = model.parameters / 1e9
        base_estimate = 65.0 + min(22.0, (p_billions ** 0.5) * 2.3)
        return AggregatedBenchmarkResult(
            composite_score=round(base_estimate, 1),
            primary_benchmark="Estimated Capability",
            primary_source="Parameter Heuristic",
            evidence_type="self_reported",
            confidence="LOW",
            has_direct_evidence=False,
            evidence_records=[],
        )

    # Process and normalize records
    has_current = any(r.get("tier", "current") == "current" for r in raw_records)
    normalized_list: List[NormalizedEvidenceRecord] = []

    total_weighted_score = 0.0
    total_weights = 0.0

    for r in raw_records:
        b_name = r["benchmark_name"]
        raw_score = float(r["score"])
        norm_score, _ = normalize_benchmark_score(b_name, raw_score)

        e_type = r.get("evidence_type") or evidence_type
        tier = r.get("tier") or "current"
        date_str = r.get("date")

        ev_weight = get_evidence_weight(e_type)
        rec_weight = compute_recency_weight(date_str, tier=tier, has_current_alternatives=has_current)
        combined_weight = round(ev_weight * rec_weight, 3)

        rec = NormalizedEvidenceRecord(
            benchmark_name=b_name,
            original_score=raw_score,
            normalized_score=norm_score,
            source=r.get("source", b_name),
            source_url=r.get("source_url"),
            date=date_str,
            evidence_type=e_type,
            tier=tier,
            weight_applied=combined_weight,
        )
        normalized_list.append(rec)

        total_weighted_score += (norm_score * combined_weight)
        total_weights += combined_weight

    composite = round(total_weighted_score / max(0.001, total_weights), 1)
    primary = normalized_list[0]

    confidence = "HIGH" if (has_direct and len(normalized_list) >= 2) else ("MEDIUM" if has_direct else "LOW")

    return AggregatedBenchmarkResult(
        composite_score=composite,
        primary_benchmark=primary.benchmark_name,
        primary_source=primary.source,
        evidence_type=evidence_type,
        confidence=confidence,
        has_direct_evidence=has_direct,
        evidence_records=normalized_list,
    )
