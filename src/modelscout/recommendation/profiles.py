"""Recommendation profiles configuring scoring weights and task prioritization."""

from enum import Enum
from typing import Dict
from pydantic import BaseModel


class RecommendationProfile(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    MATH = "math"
    VISION = "vision"
    CREATIVE = "creative"
    FAST = "fast"
    QUALITY = "quality"
    LOW_MEMORY = "low-memory"
    CPU = "cpu"


class ProfileWeights(BaseModel):
    benchmark_weight: float = 0.30
    capability_weight: float = 0.15
    fit_weight: float = 0.25
    speed_weight: float = 0.15
    evidence_weight: float = 0.10
    runtime_weight: float = 0.05
    requires_vision: bool = False
    requires_coding: bool = False
    requires_reasoning: bool = False
    min_speed_threshold: float = 0.0


def get_profile_weights(profile: str | RecommendationProfile) -> ProfileWeights:
    """Returns scoring weights and requirements for specified profile."""
    p_str = str(profile).lower()

    if p_str in ["coding", "code"]:
        return ProfileWeights(
            benchmark_weight=0.35,
            capability_weight=0.15,
            fit_weight=0.20,
            speed_weight=0.15,
            evidence_weight=0.10,
            runtime_weight=0.05,
            requires_coding=True,
        )
    elif p_str in ["reasoning", "math"]:
        return ProfileWeights(
            benchmark_weight=0.35,
            capability_weight=0.20,
            fit_weight=0.20,
            speed_weight=0.10,
            evidence_weight=0.10,
            runtime_weight=0.05,
            requires_reasoning=True,
        )
    elif p_str in ["vision", "multimodal"]:
        return ProfileWeights(
            benchmark_weight=0.25,
            capability_weight=0.20,
            fit_weight=0.25,
            speed_weight=0.15,
            evidence_weight=0.10,
            runtime_weight=0.05,
            requires_vision=True,
        )
    elif p_str in ["fast", "speed"]:
        return ProfileWeights(
            benchmark_weight=0.20,
            capability_weight=0.10,
            fit_weight=0.20,
            speed_weight=0.40,
            evidence_weight=0.05,
            runtime_weight=0.05,
            min_speed_threshold=18.0,
        )
    elif p_str in ["quality", "best"]:
        return ProfileWeights(
            benchmark_weight=0.45,
            capability_weight=0.25,
            fit_weight=0.15,
            speed_weight=0.05,
            evidence_weight=0.05,
            runtime_weight=0.05,
        )
    elif p_str in ["low-memory", "compact"]:
        return ProfileWeights(
            benchmark_weight=0.20,
            capability_weight=0.10,
            fit_weight=0.45,
            speed_weight=0.15,
            evidence_weight=0.05,
            runtime_weight=0.05,
        )
    elif p_str in ["cpu", "cpu-only"]:
        return ProfileWeights(
            benchmark_weight=0.20,
            capability_weight=0.10,
            fit_weight=0.40,
            speed_weight=0.20,
            evidence_weight=0.05,
            runtime_weight=0.05,
        )
    else:  # GENERAL
        return ProfileWeights()
