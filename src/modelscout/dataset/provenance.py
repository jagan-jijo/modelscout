"""Data provenance and freshness tracking."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class FreshnessStatus(str, Enum):
    LIVE = "LIVE"
    RECENT_CACHE = "RECENT_CACHE"
    DATASET = "DATASET"
    CURATED = "CURATED"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class Provenance(BaseModel):
    source: str
    source_url: Optional[str] = None
    retrieved_at: str
    confidence: str = "high"  # high, medium, low
    freshness: FreshnessStatus = FreshnessStatus.DATASET

    @classmethod
    def from_dataset(cls, confidence: str = "high") -> "Provenance":
        return cls(
            source="dataset.json",
            source_url="local://dataset.json",
            retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            confidence=confidence,
            freshness=FreshnessStatus.DATASET,
        )

    @classmethod
    def from_live(cls, source_name: str, url: str) -> "Provenance":
        return cls(
            source=source_name,
            source_url=url,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            confidence="high",
            freshness=FreshnessStatus.LIVE,
        )
