"""Shared utilities for CUDA test modules."""

from __future__ import annotations

import json
import sys
from typing import Any


def output_json(data: dict[str, Any]) -> None:
    """Print result as JSON to stdout and exit."""
    print(json.dumps(data, default=str))
    sys.exit(0)


def get_gpu_info(gpu_index: int) -> dict[str, Any]:
    """Get GPU name and memory via cupy."""
    try:
        import cupy as cp
        cp.cuda.Device(gpu_index).use()
        mem = cp.cuda.Device(gpu_index).mem_info
        props = cp.cuda.runtime.getDeviceProperties()
        return {
            "model": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "memory_total_mb": mem[1] // (1024 * 1024),
            "memory_free_mb": mem[0] // (1024 * 1024),
        }
    except ImportError:
        return {"model": "unknown (cupy not installed)", "memory_total_mb": 0, "memory_free_mb": 0}
