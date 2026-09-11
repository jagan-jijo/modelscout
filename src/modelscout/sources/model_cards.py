"""Utilities for parsing model cards and extracting architecture metadata."""

import re
from typing import Dict, Optional


def parse_model_card_context_length(card_text: str) -> Optional[int]:
    """Extracts context window from markdown model card text."""
    m = re.search(r"context\s*(?:window|length)?\s*(?:of|is|:)?\s*(\d+)\s*(?:k|K)", card_text)
    if m:
        return int(m.group(1)) * 1024
    return None
