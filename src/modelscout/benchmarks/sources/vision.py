"""Multimodal and Vision benchmark source connector."""

from typing import List
from modelscout.benchmarks.sources import BaseBenchmarkSource, RawBenchmarkEntry


class VisionBenchmarkSource(BaseBenchmarkSource):
    name = "Vision Benchmark"
    tier = "current"

    def fetch(self) -> List[RawBenchmarkEntry]:
        return [
            RawBenchmarkEntry(model_name="gemma-4-31b", benchmark="Vision", score=88.0, date="2026-08-28", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="gemma-4-e2b", benchmark="Vision", score=76.2, date="2026-08-28", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="mistral-small-3.1-24b", benchmark="Vision", score=81.5, date="2026-08-20", url="https://livebench.ai"),
        ]
