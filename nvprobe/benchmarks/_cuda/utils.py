"""Shared utilities for CUDA test modules."""

from __future__ import annotations

import json
import sys
from typing import Any


def require_cupy():
    """Import cupy or exit with installation instructions."""
    try:
        import cupy as cp
        return cp
    except ImportError:
        print(
            "Error: CuPy is not installed.\n"
            "Install the correct version for your CUDA toolkit:\n"
            "  CUDA 13.x: pip install cupy-cuda13x\n"
            "  CUDA 12.x: pip install cupy-cuda12x\n"
            "  CUDA 11.x: pip install cupy-cuda11x\n"
            "Or run: make install-cupy",
            file=sys.stderr,
        )
        sys.exit(1)


def output_json(data: dict[str, Any]) -> None:
    """Print result as JSON to stdout and exit."""
    print(json.dumps(data, default=str))
    sys.exit(0)


def get_gpu_info(gpu_index: int) -> dict[str, Any]:
    """Get GPU name and memory via cupy."""
    try:
        cp = require_cupy()
        cp.cuda.Device(gpu_index).use()
        mem = cp.cuda.Device(gpu_index).mem_info
        props = cp.cuda.runtime.getDeviceProperties(gpu_index)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "model": str(name),
            "memory_total_mb": mem[1] // (1024 * 1024),
            "memory_free_mb": mem[0] // (1024 * 1024),
        }
    except Exception as exc:
        return {"model": f"unknown ({exc})", "memory_total_mb": 0, "memory_free_mb": 0}
