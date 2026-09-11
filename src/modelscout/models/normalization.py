"""Model name and parameter count normalization."""

import re
from typing import Optional, Tuple


def parse_parameters_str(param_str: Optional[str]) -> Optional[int]:
    """Parses parameter strings like '7B', '0.6B', '32B', '70B' into integer count."""
    if not param_str:
        return None
    cleaned = param_str.strip().upper()
    m = re.match(r"^([\d\.]+)\s*([BMK]?)$", cleaned)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "B":
        return int(val * 1_000_000_000)
    elif unit == "M":
        return int(val * 1_000_000)
    elif unit == "K":
        return int(val * 1_000)
    return int(val)


def parse_moe_parameters(name_or_str: str) -> Tuple[Optional[int], Optional[int]]:
    """Detects total and active parameters from strings like '30b-a3b' or '235b-a22b'."""
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]-?[Aa](\d+(?:\.\d+)?)[Bb]", name_or_str)
    if m:
        total = int(float(m.group(1)) * 1_000_000_000)
        active = int(float(m.group(2)) * 1_000_000_000)
        return total, active
    return None, None


def normalize_model_name(raw_name: str) -> str:
    """Strips repo author, Instruct/chat/GGUF suffixes to obtain canonical model name."""
    name = raw_name.split("/")[-1]
    name = re.sub(r"-(GGUF|gguf|Q\d+.*)$", "", name)
    name = re.sub(r"-(Instruct|it|Chat|chat)$", "", name, flags=re.IGNORECASE)
    return name.strip()


def extract_model_family(name: str) -> str:
    """Extracts high level model family from model name."""
    lower = name.lower()
    if "deepseek-r1" in lower:
        return "DeepSeek-R1"
    elif "deepseek-v4" in lower:
        return "DeepSeek-V4"
    elif "deepseek-v3" in lower:
        return "DeepSeek-V3"
    elif "deepseek" in lower:
        return "DeepSeek"
    elif "qwen3" in lower:
        return "Qwen3"
    elif "qwen2.5-coder" in lower or "qwen2.5_coder" in lower:
        return "Qwen2.5-Coder"
    elif "qwen2.5" in lower:
        return "Qwen2.5"
    elif "qwen" in lower:
        return "Qwen"
    elif "gemma-4" in lower or "gemma4" in lower:
        return "Gemma 4"
    elif "gemma-2" in lower or "gemma2" in lower:
        return "Gemma 2"
    elif "gemma" in lower:
        return "Gemma"
    elif "mistral-small" in lower:
        return "Mistral Small"
    elif "mistral" in lower:
        return "Mistral"
    elif "phi-4" in lower or "phi4" in lower:
        return "Phi-4"
    elif "phi-3" in lower or "phi3" in lower:
        return "Phi-3"
    elif "nemotron" in lower:
        return "Nemotron"
    return name.split("-")[0].title()
