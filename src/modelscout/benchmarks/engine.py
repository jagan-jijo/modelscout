"""Benchmark engine coordinator for refreshing sources and managing aggregation."""

from typing import List, Optional
from modelscout.benchmarks.aggregation import AggregatedBenchmarkResult, aggregate_model_benchmarks
from modelscout.benchmarks.sources import BaseBenchmarkSource
from modelscout.benchmarks.sources.aider import AiderBenchmarkSource
from modelscout.benchmarks.sources.arena import ChatbotArenaSource
from modelscout.benchmarks.sources.artificial_analysis import ArtificialAnalysisSource
from modelscout.benchmarks.sources.leaderboard import OpenLLMLeaderboardSource
from modelscout.benchmarks.sources.livebench import LiveBenchSource
from modelscout.benchmarks.sources.vision import VisionBenchmarkSource
from modelscout.database.repository import DatabaseRepository
from modelscout.models.metadata import ModelMetadata


class BenchmarkEngine:
    def __init__(self, repo: Optional[DatabaseRepository] = None):
        self.repo = repo or DatabaseRepository()
        self.sources: List[BaseBenchmarkSource] = [
            LiveBenchSource(),
            ArtificialAnalysisSource(),
            AiderBenchmarkSource(),
            VisionBenchmarkSource(),
            ChatbotArenaSource(),
            OpenLLMLeaderboardSource(),
        ]

    def evaluate_model(self, model: ModelMetadata) -> AggregatedBenchmarkResult:
        """Resolves multi-source evidence and produces aggregated benchmark quality."""
        return aggregate_model_benchmarks(model, self.repo)

    def refresh_sources(self) -> int:
        """Fetches from all active benchmark sources and stores into repository."""
        conn = self.repo.get_connection()
        total_stored = 0
        try:
            cur = conn.cursor()
            for src in self.sources:
                entries = src.fetch()
                for e in entries:
                    cur.execute(
                        """INSERT OR REPLACE INTO benchmarks
                           (model_id, benchmark_name, score, normalized_score, source, source_url, date, tier)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (e.model_name, e.benchmark, e.score, e.score, src.name, e.url, e.date, e.tier),
                    )
                    total_stored += 1
            conn.commit()
            return total_stored
        finally:
            if self.repo.db_path != ":memory:":
                conn.close()
