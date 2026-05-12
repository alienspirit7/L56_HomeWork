"""Timing + MPS peak-memory helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

import torch


@contextmanager
def timed():
    """Context manager yielding a callable that returns elapsed seconds."""
    t0 = time.perf_counter()
    elapsed = {"value": 0.0}
    try:
        yield lambda: elapsed["value"]
    finally:
        elapsed["value"] = time.perf_counter() - t0


def peak_memory_mb() -> Optional[float]:
    """Return MPS driver-allocated memory in MB, or None if unavailable."""
    if not torch.backends.mps.is_available():
        return None
    try:
        bytes_used = torch.mps.driver_allocated_memory()
    except (AttributeError, RuntimeError):
        return None
    return float(bytes_used) / (1024 * 1024)
