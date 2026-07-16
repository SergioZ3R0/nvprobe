#!/usr/bin/env python3
"""Custom CUDA kernels benchmark — matmul, conv2d, attention microbenchmarks.

Usage:
    python -m nvprobe.benchmarks._cuda.custom_kernels \
        --gpu 0 --kernels matmul,conv2d,attention \
        --sizes 512,1024,2048 --iterations 50 --precision fp32 --batch-size 32

Output: JSON to stdout with per-kernel, per-size performance metrics.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from nvprobe.benchmarks._cuda.utils import get_gpu_info, output_json, require_cupy

cp = require_cupy()


def bench_matmul(gpu_index: int, sizes: list[int], iterations: int, precision: str, batch_size: int) -> dict[str, Any]:
    """Benchmark matrix multiplication."""
    cp.cuda.Device(gpu_index).use()
    dtype = cp.float32 if precision == "fp32" else cp.float16 if precision == "fp16" else cp.float32
    results: dict[str, Any] = {}

    for n in sizes:
        a = cp.ones((batch_size, n, n), dtype=dtype)
        b = cp.ones((batch_size, n, n), dtype=dtype)

        # Warmup
        for _ in range(min(5, iterations)):
            _ = a @ b
        cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iterations):
            _ = a @ b
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        avg_ms = elapsed_ms / iterations
        # GFLOPS = 2 * batch_size * n^3 / (avg_time_in_seconds * 1e9)
        gflops = (2 * batch_size * n**3) / (avg_ms / 1000 * 1e9)
        results[str(n)] = {"avg_ms": round(avg_ms, 4), "gflops": round(gflops, 2)}

    return results


def bench_conv2d(gpu_index: int, sizes: list[int], iterations: int, precision: str, batch_size: int) -> dict[str, Any]:
    """Benchmark 2D convolution."""
    from cupyx.scipy import ndimage as cp_ndimage
    cp.cuda.Device(gpu_index).use()
    dtype = cp.float32 if precision == "fp32" else cp.float16 if precision == "fp16" else cp.float32
    results: dict[str, Any] = {}

    for n in sizes:
        channels = 64
        kernel_size = 3
        inp = cp.ones((batch_size, channels, n, n), dtype=dtype)
        weight = cp.ones((channels, channels, kernel_size, kernel_size), dtype=dtype)

        # Warmup using correlation (equivalent to conv2d)
        for _ in range(min(5, iterations)):
            for b in range(batch_size):
                for c_out in range(channels):
                    cp_ndimage.correlate(inp[b, 0], weight[c_out, 0], mode="constant")
        cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iterations):
            for b in range(batch_size):
                for c_out in range(min(4, channels)):
                    cp_ndimage.correlate(inp[b, 0], weight[c_out, 0], mode="constant")
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        avg_ms = elapsed_ms / iterations
        results[str(n)] = {"avg_ms": round(avg_ms, 4), "throughput": round(batch_size * min(4, channels) * iterations / (elapsed_ms / 1000), 2)}

    return results


def bench_attention(gpu_index: int, sizes: list[int], iterations: int, precision: str, batch_size: int) -> dict[str, Any]:
    """Benchmark scaled dot-product attention: Q @ K^T / sqrt(d) @ V."""
    cp.cuda.Device(gpu_index).use()
    dtype = cp.float32 if precision == "fp32" else cp.float16 if precision == "fp16" else cp.float32
    results: dict[str, Any] = {}

    for seq_len in sizes:
        d_model = min(128, seq_len)  # head dimension
        q = cp.ones((batch_size, seq_len, d_model), dtype=dtype)
        k = cp.ones((batch_size, seq_len, d_model), dtype=dtype)
        v = cp.ones((batch_size, seq_len, d_model), dtype=dtype)
        scale = d_model ** -0.5

        # Warmup
        for _ in range(min(5, iterations)):
            scores = q @ k.transpose(0, 2, 1) * scale
            weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
            weights = weights / cp.sum(weights, axis=-1, keepdims=True)
            _ = weights @ v
        cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iterations):
            scores = q @ k.transpose(0, 2, 1) * scale
            weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
            weights = weights / cp.sum(weights, axis=-1, keepdims=True)
            _ = weights @ v
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        avg_ms = elapsed_ms / iterations
        # Approximate FLOPS: 2*B*S*D (Q@K^T) + B*S*S (softmax) + 2*B*S*D (W@V)
        flops = 2 * batch_size * seq_len * d_model + batch_size * seq_len * seq_len + 2 * batch_size * seq_len * d_model
        tflops = flops / (avg_ms / 1000 * 1e12)
        results[str(seq_len)] = {"avg_ms": round(avg_ms, 4), "tflops": round(tflops, 4)}

    return results


KERNEL_MAP = {
    "matmul": bench_matmul,
    "conv2d": bench_conv2d,
    "attention": bench_attention,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="nvProbe Custom CUDA Kernels Benchmark")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kernels", type=str, default="matmul")
    parser.add_argument("--sizes", type=str, default="512,1024,2048")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    kernels = [k.strip() for k in args.kernels.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    gpu_info = get_gpu_info(args.gpu)

    all_results: dict[str, Any] = {}
    for kernel_name in kernels:
        bench_fn = KERNEL_MAP.get(kernel_name)
        if bench_fn is None:
            all_results[kernel_name] = {"error": f"unknown kernel '{kernel_name}'"}
            continue
        all_results[kernel_name] = bench_fn(args.gpu, sizes, args.iterations, args.precision, args.batch_size)

    output_json({
        "benchmark": "custom",
        "gpu_model": gpu_info["model"],
        "gpu_index": args.gpu,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "kernels": kernels,
        "metrics": all_results,
    })


if __name__ == "__main__":
    main()
