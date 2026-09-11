"""Model family grouping and architectural relationships."""

from typing import Dict, List


KNOWN_FAMILIES: Dict[str, List[str]] = {
    "Qwen3": ["0.6B", "1.7B", "4B", "8B", "14B", "30B-A3B", "32B", "235B-A22B"],
    "Qwen2.5": ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
    "Qwen2.5-Coder": ["0.5B", "1.5B", "3B", "7B", "14B", "32B"],
    "Llama-3.3": ["70B"],
    "Llama-3.2": ["1B", "3B", "11B", "90B"],
    "Llama-3.1": ["8B", "70B", "405B"],
    "DeepSeek-R1": ["1.5B", "7B", "14B", "32B", "70B", "671B"],
    "DeepSeek-V4": ["284B-A13B"],
    "Gemma 4": ["e2B", "e4B", "12B", "26B-A4B", "31B"],
    "Gemma 2": ["2B", "9B", "27B"],
    "Mistral Small 3.1": ["24B"],
    "Phi-4": ["14B"],
}


def get_family_variants(family_name: str) -> List[str]:
    """Returns known parameter sizes within a family."""
    return KNOWN_FAMILIES.get(family_name, [])
