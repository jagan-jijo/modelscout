"""Evidence hierarchy, confidence levels, and cross-model inheritance validation."""

from enum import Enum
from typing import Optional


class EvidenceLevel(str, Enum):
    DIRECT = "direct"
    VARIANT = "variant"
    BASE_MODEL = "base_model"
    LINE_INTERPOLATED = "line_interpolated"
    SELF_REPORTED = "self_reported"


EVIDENCE_WEIGHTS = {
    EvidenceLevel.DIRECT: 1.0,
    EvidenceLevel.VARIANT: 0.95,
    EvidenceLevel.BASE_MODEL: 0.88,
    EvidenceLevel.LINE_INTERPOLATED: 0.70,
    EvidenceLevel.SELF_REPORTED: 0.50,
}


def get_evidence_weight(level: str | EvidenceLevel) -> float:
    """Returns numerical weight multiplier for evidence level."""
    if isinstance(level, EvidenceLevel):
        val = level.value
    else:
        val = str(level).lower()
        if "." in val:
            val = val.split(".")[-1]
    try:
        e = EvidenceLevel(val)
        return EVIDENCE_WEIGHTS.get(e, 0.70)
    except Exception:
        return 0.70


def validate_evidence_transfer(
    source_params: int,
    target_params: int,
    source_family: str,
    target_family: str,
) -> tuple[bool, Optional[str]]:
    """Prevents invalid benchmark inheritance across differing parameter counts or families.
    Rejects e.g. inheriting a 70B score onto a 7B model!
    """
    if source_family.lower() != target_family.lower():
        return False, f"Incompatible families: {source_family} vs {target_family}"

    ratio = max(source_params, target_params) / max(1, min(source_params, target_params))
    if ratio > 2.5:
        return (
            False,
            f"Parameter disparity too large for inheritance: {source_params / 1e9:.1f}B vs {target_params / 1e9:.1f}B (ratio {ratio:.1f}x)",
        )

    return True, None
