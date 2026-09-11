"""Open LLM Leaderboard v2 benchmark connector (frozen historical tier)."""

from typing import List
from modelscout.benchmarks.sources import BaseBenchmarkSource, RawBenchmarkEntry


class OpenLLMLeaderboardSource(BaseBenchmarkSource):
    name = "Open LLM Leaderboard v2"
    tier = "frozen"

    def fetch(self) -> List[RawBenchmarkEntry]:
        return [
            RawBenchmarkEntry(model_name="llama-3.3-70b-instruct", benchmark="Open LLM Leaderboard", score=48.5, date="2026-07-15", url="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard", tier="frozen"),
            RawBenchmarkEntry(model_name="llama-3.1-8b-instruct", benchmark="Open LLM Leaderboard", score=36.2, date="2026-07-15", url="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard", tier="frozen"),
        ]
