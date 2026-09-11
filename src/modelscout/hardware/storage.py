"""Storage abstractions and utilities."""

from modelscout.hardware.types import StorageInfo


def estimate_disk_headroom_gb(storage: StorageInfo) -> float:
    """Returns safe available disk space for model weights (leaving 15GB buffer)."""
    return max(0.0, storage.available_gb - 15.0)
