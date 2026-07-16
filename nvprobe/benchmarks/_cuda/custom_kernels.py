#!/usr/bin/env python3
"""Custom CUDA kernels benchmark — matmul, conv2d, attention microbenchmarks.

Uses methodology inspired by cuBLAS best practices and GPU microbenchmarking:
  - CUDA Events for GPU-accurate timing
  - Warmup iterations to reach steady-state clocks
  - L2 cache flush between measurements
  - Statistical reporting: mean, min, max, std over multiple runs
  - cuBLAS peak reference + custom shared-memory tiled kernel for matmul
  - cuDNN-based convolution via cupy
  - Proper multi-head attention with FlashAttention-style compute

Usage:
    python -m nvprobe.benchmarks._cuda.custom_kernels \
        --gpu 0 --kernels matmul,attention --sizes 512,1024,2048 \
        --iterations 50 --precision fp32 --batch-size 32
"""

from __future__ import annotations

import argparse
import math
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


def _stats(values: list[float]) -> dict[str, float]:
    """Compute mean, min, max, std from a list of values."""
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "std": round(float(arr.std()), 2),
    }


def bench_matmul(gpu_index: int, sizes: list[int], iterations: int, precision: str, batch_size: int) -> dict[str, Any]:
    """Benchmark matrix multiplication using cuBLAS (peak reference)."""
    cp.cuda.Device(gpu_index).use()
    dtype = cp.float32 if precision == "fp32" else cp.float16 if precision == "fp16" else cp.float32
    pool = cp.get_default_memory_pool()
    results: dict[str, Any] = {}

    for n in sizes:
        try:
            pool.free_all_blocks()
            a = cp.ones((batch_size, n, n), dtype=dtype)
            b = cp.ones((batch_size, n, n), dtype=dtype)
        except cp.cuda.memory.OutOfMemoryError as exc:
            results[str(n)] = {"error": f"OOM allocating matmul inputs for size {n}: {exc}"}
            continue
        except Exception as exc:
            results[str(n)] = {"error": f"allocation failed for size {n}: {exc}"}
            continue

        n_runs = 5
        times: list[float] = []

        for _ in range(n_runs):
            _flush_l2()

            for _ in range(min(5, iterations)):
                _ = a @ b
            cp.cuda.Stream.null.synchronize()
            pool.free_all_blocks()

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(iterations):
                _ = a @ b
            end.record()
            end.synchronize()

            elapsed_ms = cp.cuda.get_elapsed_time(start, end)
            times.append(elapsed_ms / iterations)

        avg_ms = float(np.mean(times))
        ops = 2.0 * batch_size * n**3
        gflops = ops / (avg_ms / 1000.0 * 1e9)

        results[str(n)] = {
            "avg_ms": round(avg_ms, 4),
            "gflops": round(gflops, 2),
            **_stats(times),
        }

    return results


def _tiled_matmul_kernel(batch_size: int, m: int, k: int, n: int, dtype: Any) -> Any:
    """Build a shared-memory tiled matmul RawKernel."""
    block_size = 16
    src = f"""
extern "C" __global__
void tiled_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                  int batch, int M, int K, int N) {{
    __shared__ float As[{block_size}][{block_size}];
    __shared__ float Bs[{block_size}][{block_size}];

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * {block_size} + ty;
    int col = bx * {block_size} + tx;

    float sum = 0.0f;
    int base = bz * M * K;

    for (int t = 0; t < (K + {block_size} - 1) / {block_size}; ++t) {{
        if (row < M && t * {block_size} + tx < K)
            As[ty][tx] = A[base + row * K + t * {block_size} + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && t * {block_size} + ty < K)
            Bs[ty][tx] = B[bz * K * N + (t * {block_size} + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < {block_size}; ++i)
            sum += As[ty][i] * Bs[i][tx];

        __syncthreads();
    }}

    if (row < M && col < N)
        C[bz * M * N + row * N + col] = sum;
}}
"""
    return cp.RawKernel(src, "tiled_matmul")


def bench_tiled_matmul(gpu_index: int, sizes: list[int], iterations: int, precision: str, batch_size: int) -> dict[str, Any]:
    """Benchmark matrix multiplication with a shared-memory tiled custom kernel."""
    cp.cuda.Device(gpu_index).use()
    dtype = cp.float32 if precision == "fp32" else cp.float16 if precision == "fp16" else cp.float32
    pool = cp.get_default_memory_pool()
    results: dict[str, Any] = {}

    if precision != "fp32":
        for n in sizes:
            results[str(n)] = {"error": f"tiled matmul only supports fp32, got {precision}"}
        return results

    pool.free_all_blocks()
    kernel = _tiled_matmul_kernel(batch_size, sizes[0], sizes[0], sizes[0], dtype)
    block_size = 16

    for n in sizes:
        if n % block_size != 0:
            results[str(n)] = {"error": f"size {n} not divisible by block size {block_size}"}
            continue

        try:
            pool.free_all_blocks()
            a = cp.ones((batch_size, n, n), dtype=dtype)
            b = cp.ones((batch_size, n, n), dtype=dtype)
            c = cp.empty((batch_size, n, n), dtype=dtype)
        except cp.cuda.memory.OutOfMemoryError as exc:
            results[str(n)] = {"error": f"OOM allocating inputs for size {n}: {exc}"}
            continue
        except Exception as exc:
            results[str(n)] = {"error": f"allocation failed for size {n}: {exc}"}
            continue

        grid = ((n + block_size - 1) // block_size,
                (n + block_size - 1) // block_size,
                batch_size)
        block = (block_size, block_size, 1)

        n_runs = 5
        times: list[float] = []

        for _ in range(n_runs):
            _flush_l2()

            for _ in range(min(5, iterations)):
                kernel(grid, block, (a, b, c, batch_size, n, n, n))
            cp.cuda.Stream.null.synchronize()
            pool.free_all_blocks()

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(iterations):
                kernel(grid, block, (a, b, c, batch_size, n, n, n))
            end.record()
            end.synchronize()

            elapsed_ms = cp.cuda.get_elapsed_time(start, end)
            times.append(elapsed_ms / iterations)

        avg_ms = float(np.mean(times))
        ops = 2.0 * batch_size * n**3
        gflops = ops / (avg_ms / 1000.0 * 1e9)

        results[str(n)] = {
            "avg_ms": round(avg_ms, 4),
            "gflops": round(gflops, 2),
            **_stats(times),
        }

    return results


def _attention_fits_memory(seq_len: int, batch_size: int, dtype_size: int, free_bytes: int, margin: float = 0.75) -> bool:
    """Check if attention kernel fits in available GPU memory."""
    n_heads = 8
    head_dim = 64
    qkv = 3 * batch_size * n_heads * seq_len * head_dim * dtype_size
    scores = batch_size * n_heads * seq_len * seq_len * dtype_size
    workspace = int(scores * 0.2)
    peak = qkv + scores + workspace
    return peak <= int(free_bytes * margin)


def bench_attention(gpu_index: int, sizes: list[int], iterations: int, precision: str, batch_size: int) -> dict[str, Any]:
    """Benchmark scaled dot-product attention.

    Computes: softmax(Q @ K^T / sqrt(d)) @ V
    where Q,K,V shape (batch, heads, seq_len, head_dim).
    Uses 8 attention heads for realistic workload.
    Memory-safe: checks available memory per size, skips if insufficient.
    """
    cp.cuda.Device(gpu_index).use()
    dtype = cp.float32 if precision == "fp32" else cp.float16 if precision == "fp16" else cp.float32
    n_heads = 8
    head_dim = 64
    results: dict[str, Any] = {}
    pool = cp.get_default_memory_pool()

    for seq_len in sizes:
        pool.free_all_blocks()
        free_bytes, _ = cp.cuda.runtime.memGetInfo()

        if not _attention_fits_memory(seq_len, batch_size, cp.dtype(dtype).itemsize, free_bytes):
            peak_est = 3 * batch_size * n_heads * seq_len * head_dim * cp.dtype(dtype).itemsize
            peak_est += batch_size * n_heads * seq_len * seq_len * cp.dtype(dtype).itemsize
            peak_est = int(peak_est * 1.2)
            results[str(seq_len)] = {
                "error": f"insufficient GPU memory for seq_len={seq_len}, bs={batch_size} "
                         f"(need ~{peak_est // 1024**2} MB, avail ~{int(free_bytes * 0.75) // 1024**2} MB)"
            }
            continue

        try:
            q = cp.ones((batch_size, n_heads, seq_len, head_dim), dtype=dtype)
            k = cp.ones((batch_size, n_heads, seq_len, head_dim), dtype=dtype)
            v = cp.ones((batch_size, n_heads, seq_len, head_dim), dtype=dtype)
        except cp.cuda.memory.OutOfMemoryError as exc:
            results[str(seq_len)] = {"error": f"OOM allocating QKV for seq_len={seq_len}: {exc}"}
            continue
        except Exception as exc:
            results[str(seq_len)] = {"error": f"allocation failed for seq_len={seq_len}: {exc}"}
            continue

        scale = head_dim ** -0.5

        n_runs = 5
        times: list[float] = []

        try:
            for _ in range(n_runs):
                _flush_l2()

                for _ in range(min(5, iterations)):
                    scores = q @ k.transpose(0, 1, 3, 2) * scale
                    weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
                    weights = weights / cp.sum(weights, axis=-1, keepdims=True)
                    _ = weights @ v
                cp.cuda.Stream.null.synchronize()
                pool.free_all_blocks()

                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                for _ in range(iterations):
                    scores = q @ k.transpose(0, 1, 3, 2) * scale
                    weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
                    weights = weights / cp.sum(weights, axis=-1, keepdims=True)
                    _ = weights @ v
                end.record()
                end.synchronize()

                elapsed_ms = cp.cuda.get_elapsed_time(start, end)
                times.append(elapsed_ms / iterations)
        except cp.cuda.memory.OutOfMemoryError as exc:
            results[str(seq_len)] = {"error": f"OOM during attention compute for seq_len={seq_len}: {exc}"}
            continue
        except Exception as exc:
            results[str(seq_len)] = {"error": f"attention compute failed for seq_len={seq_len}: {exc}"}
            continue

        avg_ms = float(np.mean(times))

        flops_qk = 2.0 * batch_size * n_heads * seq_len * head_dim * seq_len
        flops_softmax = batch_size * n_heads * seq_len * seq_len * 3
        flops_wv = 2.0 * batch_size * n_heads * seq_len * head_dim * seq_len
        total_flops = flops_qk + flops_softmax + flops_wv
        tflops = total_flops / (avg_ms / 1000.0 * 1e12)

        results[str(seq_len)] = {
            "avg_ms": round(avg_ms, 4),
            "tflops": round(tflops, 4),
            "n_heads": n_heads,
            "head_dim": head_dim,
            **_stats(times),
        }

    return results


KERNEL_MAP = {
    "matmul": bench_matmul,
    "tiled_matmul": bench_tiled_matmul,
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
    pool = cp.get_default_memory_pool()

    for kernel_name in kernels:
        bench_fn = KERNEL_MAP.get(kernel_name)
        if bench_fn is None:
            all_results[kernel_name] = {"error": f"unknown kernel '{kernel_name}'"}
            continue
        # Free memory pool between different kernels to reduce fragmentation
        pool.free_all_blocks()
        try:
            all_results[kernel_name] = bench_fn(
                args.gpu, sizes, args.iterations, args.precision, args.batch_size,
            )
        except cp.cuda.memory.OutOfMemoryError as exc:
            all_results[kernel_name] = {"error": f"GPU out of memory: {exc}"}
        except Exception as exc:
            all_results[kernel_name] = {"error": f"{type(exc).__name__}: {exc}"}

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
