"""Recommendation package for profiles, scoring, explanation, and ranking."""

from modelscout.recommendation.explanation import explain_why_not_recommended, explain_why_recommended
from modelscout.recommendation.profiles import ProfileWeights, RecommendationProfile, get_profile_weights
from modelscout.recommendation.ranking import (
    ExcludedCandidate,
    ModelRecommendation,
    RecommendationReport,
    generate_recommendations,
)
from modelscout.recommendation.scoring import ScoreBreakdown, calculate_recommendation_score

__all__ = [
    "RecommendationProfile",
    "ProfileWeights",
    "get_profile_weights",
    "ScoreBreakdown",
    "calculate_recommendation_score",
    "explain_why_recommended",
    "explain_why_not_recommended",
    "ModelRecommendation",
    "ExcludedCandidate",
    "RecommendationReport",
    "generate_recommendations",
]
