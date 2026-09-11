"""KV cache memory calculation accounting for GQA, context length, and precision."""

from typing import Optional


def calculate_kv_cache_bytes(
    context_tokens: int,
    num_layers: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    bytes_per_element: float = 2.0,  # FP16 = 2 bytes, FP8 = 1 byte
    batch_size: int = 1,
) -> int:
    """Calculates KV cache memory in bytes: 2 * layers * kv_heads * head_dim * context * bytes_per_element * batch_size."""
    return int(2 * num_layers * kv_heads * head_dim * context_tokens * bytes_per_element * batch_size)


def estimate_kv_cache_from_params(
    parameters: int,
    context_tokens: int,
    is_gqa: bool = True,
    bytes_per_element: float = 2.0,
) -> int:
    """Heuristic KV cache calculation when exact layers/kv_heads are not in metadata."""
    # Approximate based on parameter size
    if parameters >= 60_000_000_000:
        layers = 80
        kv_heads = 8 if is_gqa else 64
        head_dim = 128
    elif parameters >= 25_000_000_000:
        layers = 64
        kv_heads = 8 if is_gqa else 40
        head_dim = 128
    elif parameters >= 12_000_000_000:
        layers = 48
        kv_heads = 8 if is_gqa else 32
        head_dim = 128
    elif parameters >= 6_000_000_000:
        layers = 32
        kv_heads = 8 if is_gqa else 32
        head_dim = 128
    else:
        layers = 24
        kv_heads = 4 if is_gqa else 16
        head_dim = 64

    return calculate_kv_cache_bytes(
        context_tokens=context_tokens,
        num_layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
    )
