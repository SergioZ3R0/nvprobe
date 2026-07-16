#!/usr/bin/env python3
"""Memory bandwidth benchmark — measures host↔device and device↔device transfer rates.

Uses methodology inspired by NVIDIA NVbandwidth and cuda-samples/bandwidthTest:
  - CUDA Events for GPU-accurate timing
  - Warmup iterations to reach steady-state clocks
  - L2 cache flush between measurements
  - Statistical reporting: mean, min, max, std over multiple runs
  - Pointer-chase latency measurement
  - Bidirectional bandwidth via concurrent streams

Usage:
    python -m nvprobe.benchmarks._cuda.bandwidth_test \
        --gpu 0 --sizes 1,4,16,64,256 --iterations 100 --precision fp32
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

from nvprobe.benchmarks._cuda.utils import get_gpu_info, output_json, require_cupy

cp = require_cupy()


def _flush_l2() -> None:
    """Clear L2 cache by writing to a large temporary buffer."""
    buf = cp.empty(256 * 1024 * 1024, dtype=cp.uint8)  # 256 MB
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


def run_bandwidth_test(
    gpu_index: int,
    sizes_mb: list[int],
    iterations: int,
    precision: str,
) -> dict[str, Any]:
    """Run bandwidth test with statistical reporting across multiple runs."""
    cp.cuda.Device(gpu_index).use()

    dtype_map = {
        "fp32": cp.float32,
        "fp16": cp.float16,
        "int8": cp.int8,
    }
    cupy_dtype = cp.dtype(dtype_map.get(precision, cp.float32))
    np_dtype_map = {
        "fp32": np.float32,
        "fp16": np.float16,
        "int8": np.int8,
    }
    numpy_dtype = np_dtype_map.get(precision, np.float32)

    n_runs = 5   # number of independent runs for statistics
    results: dict[str, Any] = {"h2d": {}, "d2h": {}, "d2d": {}, "bidir": {}, "latency_ns": {}}

    for size_mb in sizes_mb:
        n_bytes = size_mb * 1024 * 1024
        n_elements = n_bytes // cupy_dtype.itemsize

        # Allocate host and device buffers
        host_data = np.ones(n_elements, dtype=numpy_dtype)
        device_data = cp.empty(n_elements, dtype=cupy_dtype)
        device_data2 = cp.empty(n_elements, dtype=cupy_dtype)

        h2d_times: list[float] = []
        d2h_times: list[float] = []
        d2d_times: list[float] = []

        for _ in range(n_runs):
            _flush_l2()

            # Warmup
            for _ in range(min(10, iterations)):
                device_data.set(host_data)
                _ = device_data.get()
            cp.cuda.Stream.null.synchronize()

            # Host → Device
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(iterations):
                device_data.set(host_data)
            end.record()
            end.synchronize()
            h2d_times.append(cp.cuda.get_elapsed_time(start, end))

            _flush_l2()

            # Device → Host
            start.record()
            for _ in range(iterations):
                host_data[:] = device_data.get()
            end.record()
            end.synchronize()
            d2h_times.append(cp.cuda.get_elapsed_time(start, end))

            _flush_l2()

            # Device → Device
            start.record()
            for _ in range(iterations):
                cp.copyto(device_data2, device_data)
            end.record()
            end.synchronize()
            d2d_times.append(cp.cuda.get_elapsed_time(start, end))

        # Bandwidth = total bytes / total time
        total_bytes = n_bytes * iterations
        results["h2d"][str(size_mb)] = {
            **_stats([_bandwidth_gb(total_bytes, t) for t in h2d_times]),
            "avg_ms": round(np.mean(h2d_times) / iterations, 4),
        }
        results["d2h"][str(size_mb)] = {
            **_stats([_bandwidth_gb(total_bytes, t) for t in d2h_times]),
            "avg_ms": round(np.mean(d2h_times) / iterations, 4),
        }
        results["d2d"][str(size_mb)] = {
            **_stats([_bandwidth_gb(total_bytes, t) for t in d2d_times]),
            "avg_ms": round(np.mean(d2d_times) / iterations, 4),
        }

        # Bidirectional bandwidth: concurrent H2D + D2H via streams
        bidir_times: list[float] = []
        for _ in range(n_runs):
            _flush_l2()
            stream_h2d = cp.cuda.Stream()
            stream_d2h = cp.cuda.Stream()

            start.record()
            for _ in range(iterations):
                with stream_h2d:
                    device_data.set(host_data)
                with stream_d2h:
                    host_data[:] = device_data.get()
            end.record()
            end.synchronize()
            bidir_times.append(cp.cuda.get_elapsed_time(start, end))

        total_bytes_bidir = 2 * n_bytes * iterations
        results["bidir"][str(size_mb)] = {
            **_stats([_bandwidth_gb(total_bytes_bidir, t) for t in bidir_times]),
            "avg_ms": round(np.mean(bidir_times) / iterations, 4),
        }

        # Pointer-chase latency measurement
        latency_ns = _measure_latency(gpu_index, n_elements, n_runs)
        results["latency_ns"][str(size_mb)] = latency_ns

    return results


def _measure_latency(gpu_index: int, n_elements: int, n_runs: int) -> dict[str, float]:
    """Measure global memory latency via pointer chasing.

    Creates a strided linked list in device memory and follows pointers
    to measure random-access latency. Methodology from NVbandwidth and
    classic P-chase microbenchmarks.
    """
    cp.cuda.Device(gpu_index).use()

    all_latencies: list[float] = []

    for _ in range(n_runs):
        _flush_l2()

        stride = 256
        indices_size = min(n_elements, 1024 * 1024)  # cap at 1M elements
        if indices_size < stride:
            continue

        # Build chase chain: indices[i] = (i + stride) % indices_size
        indices_host = np.arange(indices_size, dtype=np.int32)
        indices_device = cp.array(indices_host)

        # Warmup: follow chain a few times
        idx = 0
        for _ in range(min(indices_size // stride, 100)):
            idx = int(indices_device[idx % indices_size].get())
        cp.cuda.Stream.null.synchronize()

        # Timed pointer chase
        n_chase = min(10000, indices_size // stride)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        idx = 0
        for _ in range(n_chase):
            idx = int(indices_device[idx % indices_size].get())
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        latency_ns = (elapsed_ms * 1e6) / n_chase  # ns per access
        all_latencies.append(latency_ns)

    if not all_latencies:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return _stats(all_latencies)


def main() -> None:
    parser = argparse.ArgumentParser(description="nvProbe Bandwidth Test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sizes", type=str, default="1,4,16,64,256")
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
