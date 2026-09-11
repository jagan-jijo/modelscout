"""Benchmark recency weighting and frozen tier demotion."""

from datetime import datetime, timezone
from typing import Optional


def compute_recency_weight(
    date_str: Optional[str],
    tier: str = "current",
    has_current_alternatives: bool = False,
) -> float:
    """Calculates recency weight multiplier. Frozen sources are demoted if current evidence is present."""
    base_weight = 1.0

    # Frozen tier demotion
    if tier.lower() == "frozen":
        base_weight = 0.85 if not has_current_alternatives else 0.70

    if not date_str:
        return base_weight * 0.90

    try:
        b_date = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - b_date).days
        if age_days < 90:
            age_multiplier = 1.0
        elif age_days < 180:
            age_multiplier = 0.95
        elif age_days < 365:
            age_multiplier = 0.88
        else:
            age_multiplier = 0.75
    except Exception:
        age_multiplier = 0.90

    return round(base_weight * age_multiplier, 3)
