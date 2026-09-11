"""RAM and disk space detection for ModelScout."""

from __future__ import annotations

import os
import shutil
import psutil

from modelscout.hardware.types import MemoryInfo


def format_bytes_human(b: int) -> str:
    """Formats bytes into human readable string."""
    gb = b / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = b / (1024**2)
    return f"{mb:.0f} MB"


def detect_ram_bytes() -> int:
    """Get total physical RAM in bytes."""
    return psutil.virtual_memory().total


def detect_available_ram_bytes() -> int:
    """Get currently available RAM bytes."""
    return psutil.virtual_memory().available


def estimate_usable_ram(total: int) -> int:
    """Estimate RAM available for model loading after OS/background reserve.

    Uses a bounded-reserve formula: total - clamp(total * 0.15, 4 GiB, 32 GiB).
    """
    _GiB = 1024**3
    reserve = int(total * 0.15)
    reserve = max(4 * _GiB, min(reserve, 32 * _GiB))
    return max(0, total - reserve)


def effective_usable_ram(total: int, budget: int | None = None) -> int:
    """Estimate usable RAM, optionally capped by a user/runtime budget."""
    usable = estimate_usable_ram(total)
    if budget is None:
        return usable
    return max(0, min(usable, budget))


def detect_disk_free_bytes(path: str | None = None) -> int:
    """Get free disk space in bytes at the given path.

    Defaults to the user's home directory, which is more accurate
    on macOS where / may be a read-only system volume.
    """
    if path is None:
        path = os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except OSError:
        return 0
