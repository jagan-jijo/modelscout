"""Benchmark source modules."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RawBenchmarkEntry(BaseModel):
    model_name: str
    benchmark: str
    score: float
    date: Optional[str] = None
    url: Optional[str] = None
    tier: str = "current"


class BaseBenchmarkSource:
    name: str = "Base"
    tier: str = "current"

    def fetch(self) -> List[RawBenchmarkEntry]:
        """Fetches latest benchmark data or returns curated snapshot."""
        return []
