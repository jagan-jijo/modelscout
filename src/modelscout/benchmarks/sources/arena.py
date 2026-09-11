"""Chatbot Arena / LMSYS ELO benchmark connector (frozen historical tier)."""

from typing import List
from modelscout.benchmarks.sources import BaseBenchmarkSource, RawBenchmarkEntry


class ChatbotArenaSource(BaseBenchmarkSource):
    name = "Chatbot Arena"
    tier = "frozen"

    def fetch(self) -> List[RawBenchmarkEntry]:
        # ELO scores around 1200-1350
        return [
            RawBenchmarkEntry(model_name="llama-3.3-70b-instruct", benchmark="Chatbot Arena", score=1310.0, date="2026-08-01", url="https://chat.lmsys.org", tier="frozen"),
            RawBenchmarkEntry(model_name="llama-3.1-8b-instruct", benchmark="Chatbot Arena", score=1205.0, date="2026-08-01", url="https://chat.lmsys.org", tier="frozen"),
            RawBenchmarkEntry(model_name="qwen2.5-coder-32b-instruct", benchmark="Chatbot Arena", score=1285.0, date="2026-08-01", url="https://chat.lmsys.org", tier="frozen"),
        ]
