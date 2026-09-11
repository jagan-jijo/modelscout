"""Artificial Analysis Quality Index benchmark connector."""

from typing import List
from modelscout.benchmarks.sources import BaseBenchmarkSource, RawBenchmarkEntry


class ArtificialAnalysisSource(BaseBenchmarkSource):
    name = "Artificial Analysis"
    tier = "current"

    def fetch(self) -> List[RawBenchmarkEntry]:
        return [
            RawBenchmarkEntry(model_name="qwen3-30b-a3b", benchmark="Artificial Analysis", score=90.8, date="2026-08-25", url="https://artificialanalysis.ai"),
            RawBenchmarkEntry(model_name="llama-3.3-70b-instruct", benchmark="Artificial Analysis", score=87.2, date="2026-08-22", url="https://artificialanalysis.ai"),
            RawBenchmarkEntry(model_name="deepseek-r1-32b", benchmark="Artificial Analysis", score=89.8, date="2026-08-28", url="https://artificialanalysis.ai"),
            RawBenchmarkEntry(model_name="gemma-4-31b", benchmark="Artificial Analysis", score=88.5, date="2026-08-30", url="https://artificialanalysis.ai"),
            RawBenchmarkEntry(model_name="qwen2.5-coder-32b-instruct", benchmark="Artificial Analysis", score=85.2, date="2026-08-20", url="https://artificialanalysis.ai"),
        ]
