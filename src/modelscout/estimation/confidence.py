"""Confidence breakdown across hardware, metadata, benchmarks, and performance."""

from typing import Dict
from pydantic import BaseModel


class ConfidenceBreakdown(BaseModel):
    overall_confidence: str  # HIGH, MEDIUM, LOW
    hardware_detection: str = "HIGH"
    model_metadata: str = "HIGH"
    benchmark_evidence: str = "HIGH"
    memory_estimate: str = "HIGH"
    speed_estimate: str = "MEDIUM"
    runtime_compatibility: str = "HIGH"

    @classmethod
    def create(
        cls,
        benchmark_conf: str = "HIGH",
        speed_conf: str = "MEDIUM",
        runtime_conf: str = "HIGH",
    ) -> "ConfidenceBreakdown":
        scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        avg = (scores.get(benchmark_conf, 2) + scores.get(speed_conf, 2) + scores.get(runtime_conf, 3)) / 3.0
        overall = "HIGH" if avg >= 2.6 else ("MEDIUM" if avg >= 1.8 else "LOW")
        return cls(
            overall_confidence=overall,
            benchmark_evidence=benchmark_conf,
            speed_estimate=speed_conf,
            runtime_compatibility=runtime_conf,
        )
