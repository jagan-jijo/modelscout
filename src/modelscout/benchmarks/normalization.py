"""Benchmark score normalization across different scales and metrics into a unified 0-100 score."""

from typing import Tuple


def normalize_benchmark_score(benchmark_name: str, raw_score: float) -> Tuple[float, str]:
    """Normalizes raw benchmark score to a 0-100 scale. Returns (normalized_score, method)."""
    b_lower = benchmark_name.lower()

    if "arena" in b_lower or "elo" in b_lower or "lmsys" in b_lower:
        # Chatbot Arena ELO typically ranges from ~950 (weak) to ~1450 (state of the art)
        # Min: 950 -> 0, Max: 1450 -> 100
        clamped = max(950.0, min(raw_score, 1450.0))
        normalized = (clamped - 950.0) / (1450.0 - 950.0) * 100.0
        return round(normalized, 1), "elo_linear_950_1450"

    elif "livebench" in b_lower or "artificial analysis" in b_lower or "aider" in b_lower or "vision" in b_lower:
        # Already 0-100 percentage scale
        return round(max(0.0, min(raw_score, 100.0)), 1), "percentage_identity"

    elif "leaderboard" in b_lower:
        # Open LLM Leaderboard v2 average is 0-100
        return round(max(0.0, min(raw_score, 100.0)), 1), "leaderboard_percentage"

    # Default fallback
    if raw_score <= 1.0:
        return round(raw_score * 100.0, 1), "ratio_to_100"
    return round(max(0.0, min(raw_score, 100.0)), 1), "default_clamp_100"
