#!/usr/bin/env python3
"""STREAM memory bandwidth benchmark — measures sustainable bandwidth via copy/scale/add/triad.

Implements the STREAM benchmark methodology (John D. McCalpin):
  - COPY:   a[i] = b[i]
  - SCALE:  a[i] = q * b[i]
  - ADD:    a[i] = b[i] + c[i]
  - TRIAD:  a[i] = b[i] + q * c[i]

Uses CUDA Events for GPU-accurate timing, warmup iterations, and
statistical reporting (mean, min, max, std) across multiple runs.

Usage:
    python -m nvprobe.benchmarks._cuda.stream_test \
        --gpu 0 --sizes 10000000,100000000 --iterations 20 --precision fp32
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from nvprobe.benchmarks._cuda.utils import get_gpu_info, output_json, require_cupy

cp = require_cupy()


def _flush_l2() -> None:
    """Clear L2 cache by writing to a large temporary buffer."""
    buf = cp.empty(256 * 1024 * 1024, dtype=cp.uint8)
    buf.fill(0)
    cp.cuda.Stream.null.synchronize()
    del buf


def _bandwidth_gb(total_bytes: float, elapsed_ms: float) -> float:
    """Compute bandwidth in GB/s from total bytes and elapsed milliseconds."""
    if elapsed_ms <= 0:
        return 0.0
    return (total_bytes / (elapsed_ms / 1000.0)) / 1e9


def _stats(values: list[float]) -> dict[str, float]:
    """Compute mean, min, max, std from a list of values."""
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "std": round(float(arr.std()), 2),
    }


def run_stream_test(
    gpu_index: int,
    sizes: list[int],
    iterations: int,
    precision: str,
) -> dict[str, Any]:
    """Run STREAM benchmark with statistical reporting across multiple runs."""
    cp.cuda.Device(gpu_index).use()

    dtype_map = {
        "fp32": cp.float32,
        "fp64": cp.float64,
        "fp16": cp.float16,
    }
    cupy_dtype = cp.dtype(dtype_map.get(precision, cp.float32))

    n_runs = 5
    scalar = cp.float64(3.0) if precision == "fp64" else cp.float32(3.0)
    results: dict[str, Any] = {}

    for n_elements in sizes:
        n_bytes_per_element = cupy_dtype.itemsize
        total_bytes = n_elements * n_bytes_per_element

        a = cp.empty(n_elements, dtype=cupy_dtype)
        b = cp.empty(n_elements, dtype=cupy_dtype)
        c = cp.empty(n_elements, dtype=cupy_dtype)

        b.fill(1.0)
        c.fill(2.0)

        copy_times: list[float] = []
        scale_times: list[float] = []
        add_times: list[float] = []
        triad_times: list[float] = []

        for _ in range(n_runs):
            _flush_l2()

            # Warmup
            for _ in range(min(10, iterations)):
                a[:] = b[:]
            cp.cuda.Stream.null.synchronize()

            # COPY: a[i] = b[i]
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(iterations):
                a[:] = b[:]
            end.record()
            end.synchronize()
            copy_times.append(cp.cuda.get_elapsed_time(start, end))

            _flush_l2()

            # SCALE: a[i] = q * b[i]
            start.record()
            for _ in range(iterations):
                a[:] = scalar * b[:]
            end.record()
            end.synchronize()
            scale_times.append(cp.cuda.get_elapsed_time(start, end))

            _flush_l2()

            # ADD: a[i] = b[i] + c[i]
            start.record()
            for _ in range(iterations):
                a[:] = b[:] + c[:]
            end.record()
            end.synchronize()
            add_times.append(cp.cuda.get_elapsed_time(start, end))

            _flush_l2()

            # TRIAD: a[i] = b[i] + q * c[i]
            start.record()
            for _ in range(iterations):
                a[:] = b[:] + scalar * c[:]
            end.record()
            end.synchronize()
            triad_times.append(cp.cuda.get_elapsed_time(start, end))

        # Each operation reads/writes arrays differently:
        # COPY:   1 read + 1 write = 2 * N * sizeof
        # SCALE:  1 read + 1 write = 2 * N * sizeof
        # ADD:    2 reads + 1 write = 3 * N * sizeof
        # TRIAD:  2 reads + 1 write = 3 * N * sizeof
        copy_bytes = 2 * total_bytes * iterations
        scale_bytes = 2 * total_bytes * iterations
        add_bytes = 3 * total_bytes * iterations
        triad_bytes = 3 * total_bytes * iterations

        results[str(n_elements)] = {
            "copy": {
                **_stats([_bandwidth_gb(copy_bytes, t) for t in copy_times]),
                "avg_ms": round(float(np.mean(copy_times)) / iterations, 4),
            },
            "scale": {
                **_stats([_bandwidth_gb(scale_bytes, t) for t in scale_times]),
                "avg_ms": round(float(np.mean(scale_times)) / iterations, 4),
            },
            "add": {
                **_stats([_bandwidth_gb(add_bytes, t) for t in add_times]),
                "avg_ms": round(float(np.mean(add_times)) / iterations, 4),
            },
            "triad": {
                **_stats([_bandwidth_gb(triad_bytes, t) for t in triad_times]),
                "avg_ms": round(float(np.mean(triad_times)) / iterations, 4),
            },
            "n_elements": n_elements,
            "total_bytes_mb": round(total_bytes / (1024 * 1024), 2),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="nvProbe STREAM Benchmark")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--sizes",
        type=str,
        default="10000000,100000000",
        help="Comma-separated element counts",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--precision", type=str, default="fp32")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    gpu_info = get_gpu_info(args.gpu)
    stream_results = run_stream_test(args.gpu, sizes, args.iterations, args.precision)

    output_json(
        {
            "benchmark": "stream",
            "gpu_model": gpu_info["model"],
            "gpu_index": args.gpu,
            "precision": args.precision,
            "iterations": args.iterations,
            "sizes": sizes,
            "metrics": stream_results,
        }
    )


if __name__ == "__main__":
    main()
