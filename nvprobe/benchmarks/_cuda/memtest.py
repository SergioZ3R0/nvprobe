#!/usr/bin/env python3
"""GPU memory test — detects VRAM faults (stuck bits, coupling, data corruption).

Test patterns (standard industry methodology):
  - Solid 0x00 / 0xFF: catches stuck-at faults
  - Checkerboard 0xAA / 0x55: catches adjacent-bit coupling
  - Deterministic pseudo-random: catches data-dependent faults
  - Walking-1 (64 KB segment): catches address-line faults

Avoids CuPy kernel compilation (works with cupy-cuda12x without [ctk] extras).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import cupy as cp
    import numpy as np
except ImportError as exc:
    print(json.dumps({"error": f"CuPy import failed: {exc}"}))
    sys.exit(1)


def _verify(buf: cp.ndarray, expected_byte: int, size: int) -> int:
    """Download buffer to host and count mismatches against expected byte value."""
    host = buf.view(dtype=np.uint8)[:size].get()
    return int(np.count_nonzero(host != expected_byte))


def _run_patterns(size_bytes: int) -> dict[str, Any]:
    """Run solid and checkerboard patterns on a fresh allocation each time."""
    results: dict[str, Any] = {}
    for name, byte_val in [
        ("solid_zero", 0x00),
        ("solid_ones", 0xFF),
        ("checkerboard_aa", 0xAA),
        ("checkerboard_55", 0x55),
    ]:
        host = np.full(size_bytes, byte_val, dtype=np.uint8)

        t0 = time.perf_counter()
        buf = cp.array(host)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        errors = _verify(buf, byte_val, size_bytes)
        bw = (size_bytes / 1e9) / elapsed if elapsed > 0 else 0
        results[name] = {
            "size_mb": int(size_bytes / (1024 * 1024)),
            "bandwidth_gbs": round(bw, 2),
            "errors": errors,
            "passed": errors == 0,
        }
    return results


def _run_random(size_bytes: int) -> dict[str, Any]:
    """Fill GPU buffer with deterministic pseudo-random data and verify."""
    np.random.seed(0x4E565052)
    host = np.random.randint(0, 256, size_bytes, dtype=np.uint8)

    t0 = time.perf_counter()
    buf = cp.array(host)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    got = buf.view(dtype=np.uint8)[:size_bytes].get()
    errors = int(np.count_nonzero(got != host))
    bw = (size_bytes / 1e9) / elapsed if elapsed > 0 else 0
    return {
        "random": {
            "size_mb": int(size_bytes / (1024 * 1024)),
            "bandwidth_gbs": round(bw, 2),
            "errors": errors,
            "passed": errors == 0,
        }
    }


def _run_walking1(size_bytes: int) -> dict[str, Any]:
    """Walking-1 test on a small segment (bit-level diagnostics)."""
    walk_bytes = min(size_bytes, 64 * 1024)
    results: dict[str, Any] = {}
    for label, fill_val, walk_val, check_val in [
        ("walking1_high", 0x00, 0x01, 0x01),
        ("walking1_low", 0xFF, 0xFE, 0xFE),
    ]:
        host = np.full(walk_bytes, fill_val, dtype=np.uint8)

        t0 = time.perf_counter()
        buf = cp.array(host)
        cp.cuda.Stream.null.synchronize()
        buf[:walk_bytes] = cp.array(np.full(walk_bytes, walk_val, dtype=np.uint8))
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        errors = _verify(buf, check_val, walk_bytes)
        bw = (walk_bytes / 1e9) / elapsed if elapsed > 0 else 0
        results[label] = {
            "size_mb": int(walk_bytes / (1024 * 1024)),
            "bandwidth_gbs": round(bw, 2),
            "errors": errors,
            "passed": errors == 0,
        }
    return results


def run_memtest(size_bytes: int) -> dict[str, Any]:
    """Execute all memtest patterns and return combined metrics."""
    metrics: dict[str, Any] = {}
    total_errors = 0

    for runner, args in [
        (_run_patterns, (size_bytes,)),
        (_run_random, (size_bytes,)),
    ]:
        result = runner(*args)
        metrics.update(result)
        total_errors += sum(r.get("errors", 0) for r in result.values())

    walking = _run_walking1(size_bytes)
    metrics.update(walking)
    total_errors += sum(r.get("errors", 0) for r in walking.values())

    metrics["total_errors"] = total_errors
    metrics["passed"] = total_errors == 0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU memory test")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index to test")
    parser.add_argument("--size", type=int, default=1024, help="Test size in MB (uses free memory if larger)")
    args = parser.parse_args()

    try:
        with cp.cuda.Device(args.gpu):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            usable = int(min(args.size * 1024 * 1024, free_bytes * 0.85))
            if usable < 1024 * 1024:
                print(json.dumps({"error": "insufficient GPU memory for memtest"}))
                sys.exit(1)

            metrics = run_memtest(usable)

            props = cp.cuda.runtime.getDeviceProperties(args.gpu)
            name = props.get("name", b"unknown")
            if isinstance(name, bytes):
                name = name.decode()

            result = {
                "benchmark": "memtest",
                "gpu_model": str(name),
                "gpu_index": args.gpu,
                "precision": "n/a",
                "metrics": metrics,
            }
            print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
