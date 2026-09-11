"""LiveBench benchmark source connector."""

from typing import List
from modelscout.benchmarks.sources import BaseBenchmarkSource, RawBenchmarkEntry


class LiveBenchSource(BaseBenchmarkSource):
    name = "LiveBench"
    tier = "current"

    def fetch(self) -> List[RawBenchmarkEntry]:
        # Curated snapshot updated 2026-08/09
        return [
            RawBenchmarkEntry(model_name="qwen3-30b-a3b", benchmark="LiveBench", score=91.4, date="2026-08-21", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="llama-3.3-70b-instruct", benchmark="LiveBench", score=88.6, date="2026-08-20", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="deepseek-r1-32b", benchmark="LiveBench", score=90.3, date="2026-08-25", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="deepseek-r1-14b", benchmark="LiveBench", score=87.1, date="2026-08-25", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="deepseek-r1-7b", benchmark="LiveBench", score=83.9, date="2026-08-25", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="gemma-4-31b", benchmark="LiveBench", score=89.2, date="2026-08-28", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="qwen2.5-coder-32b-instruct", benchmark="LiveBench", score=86.4, date="2026-08-18", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="phi-4-14b", benchmark="LiveBench", score=86.7, date="2026-08-10", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="mistral-small-3.1-24b", benchmark="LiveBench", score=85.0, date="2026-08-20", url="https://livebench.ai"),
            RawBenchmarkEntry(model_name="llama-3.1-8b-instruct", benchmark="LiveBench", score=73.5, date="2026-08-10", url="https://livebench.ai"),
        ]
