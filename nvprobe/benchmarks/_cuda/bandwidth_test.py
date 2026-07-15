#!/usr/bin/env python3
"""Memory bandwidth benchmark — measures host↔device and device↔device transfer rates.

Usage:
    python -m nvprobe.benchmarks._cuda.bandwidth_test \
        --gpu 0 --sizes 1,4,16,64,256,1024 --iterations 100 --precision fp32

Output: JSON to stdout with per-size bandwidth measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import cupy as cp

from nvprobe.benchmarks._cuda.utils import get_gpu_info, output_json


def run_bandwidth_test(
    gpu_index: int,
    sizes_mb: list[int],
    iterations: int,
    precision: str,
) -> dict[str, Any]:
    """Run bandwidth test across specified buffer sizes."""
    cp.cuda.Device(gpu_index).use()

    dtype_map = {
        "fp32": cp.float32,
        "fp16": cp.float16,
        "int8": cp.int8,
    }
    dtype = dtype_map.get(precision, cp.float32)

    results: dict[str, Any] = {"h2d": {}, "d2h": {}, "d2d": {}}

    for size_mb in sizes_mb:
        n_elements = (size_mb * 1024 * 1024) // dtype.itemsize
        host_data = cp.ones(n_elements, dtype=dtype)
        device_data = cp.empty(n_elements, dtype=dtype)

        # Warmup
        for _ in range(min(10, iterations)):
            device_data.copy_from(host_data)
            cp.cuda.Stream.null.synchronize()

        # Host → Device
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iterations):
            device_data.copy_from(host_data)
        end.record()
        end.synchronize()
        h2d_ms = cp.cuda.get_elapsed_time(start, end)
        h2d_bw = (size_mb * iterations) / (h2d_ms / 1000)  # MB/s
        results["h2d"][str(size_mb)] = round(h2d_bw, 2)

        # Device → Host
        start.record()
        for _ in range(iterations):
            host_data.copy_from(device_data)
        end.record()
        end.synchronize()
        d2h_ms = cp.cuda.get_elapsed_time(start, end)
        d2h_bw = (size_mb * iterations) / (d2h_ms / 1000)
        results["d2h"][str(size_mb)] = round(d2h_bw, 2)

        # Device → Device
        device_data2 = cp.empty(n_elements, dtype=dtype)
        start.record()
        for _ in range(iterations):
            device_data2.copy_from(device_data)
        end.record()
        end.synchronize()
        d2d_ms = cp.cuda.get_elapsed_time(start, end)
        d2d_bw = (size_mb * iterations) / (d2d_ms / 1000)
        results["d2d"][str(size_mb)] = round(d2d_bw, 2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="nvProbe Bandwidth Test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sizes", type=str, default="1,4,16,64,256,1024")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--precision", type=str, default="fp32")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    gpu_info = get_gpu_info(args.gpu)
    bw_results = run_bandwidth_test(args.gpu, sizes, args.iterations, args.precision)

    output_json({
        "benchmark": "bandwidth",
        "gpu_model": gpu_info["model"],
        "gpu_index": args.gpu,
        "precision": args.precision,
        "iterations": args.iterations,
        "sizes_mb": sizes,
        "metrics": bw_results,
    })


if __name__ == "__main__":
    main()
