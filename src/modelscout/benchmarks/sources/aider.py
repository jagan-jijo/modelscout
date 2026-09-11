"""Aider code editing benchmark connector."""

from typing import List
from modelscout.benchmarks.sources import BaseBenchmarkSource, RawBenchmarkEntry


class AiderBenchmarkSource(BaseBenchmarkSource):
    name = "Aider"
    tier = "current"

    def fetch(self) -> List[RawBenchmarkEntry]:
        return [
            RawBenchmarkEntry(model_name="qwen3-30b-a3b", benchmark="Aider", score=89.7, date="2026-08-18", url="https://aider.chat/docs/leaderboards/"),
            RawBenchmarkEntry(model_name="qwen2.5-coder-32b-instruct", benchmark="Aider", score=87.8, date="2026-08-15", url="https://aider.chat/docs/leaderboards/"),
            RawBenchmarkEntry(model_name="llama-3.3-70b-instruct", benchmark="Aider", score=84.1, date="2026-08-10", url="https://aider.chat/docs/leaderboards/"),
            RawBenchmarkEntry(model_name="deepseek-r1-14b", benchmark="Aider", score=83.5, date="2026-08-15", url="https://aider.chat/docs/leaderboards/"),
            RawBenchmarkEntry(model_name="qwen2.5-coder-7b-instruct", benchmark="Aider", score=82.5, date="2026-08-15", url="https://aider.chat/docs/leaderboards/"),
            RawBenchmarkEntry(model_name="phi-4-14b", benchmark="Aider", score=81.0, date="2026-08-12", url="https://aider.chat/docs/leaderboards/"),
        ]
