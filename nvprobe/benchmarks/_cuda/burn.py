#!/usr/bin/env python3
"""Sustained GPU compute burn — detects thermal/power throttling over time.

Runs a dense matmul (cuBLAS SGEMM) in a tight loop for the configured duration
while a background thread samples GPU clocks, temperature, and power draw via
nvidia-smi. Produces time-series metrics for throttling analysis.

Usage:
    python -m nvprobe.benchmarks._cuda.burn --gpu 0 --duration 30 --size 8192
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from typing import Any

import cupy as cp
import numpy as np


def _sample_clocks(
    gpu_index: int,
    samples: list[dict[str, Any]],
    start_time: float,
    stop_event: threading.Event,
    interval: float = 1.0,
) -> None:
    """Background thread: sample GPU clocks/temp/power via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
        "-i", str(gpu_index),
    ]
    while not stop_event.is_set():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            if len(parts) >= 4:
                samples.append({
                    "t": round(time.perf_counter() - start_time, 1),
                    "sm": _parse_or_none(parts[0]),
                    "mem": _parse_or_none(parts[1]),
                    "temp": _parse_or_none(parts[2]),
                    "power": _parse_or_none(parts[3]),
                })
        except Exception:
            pass
        time.sleep(interval)


def _parse_or_none(value: str) -> int | None:
    try:
        v = value.strip()
        if v in ("", "N/A", "Unknown", "[Not Supported]", "[N/A]"):
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def run_burn(gpu_index: int, size: int, duration_sec: int) -> dict[str, Any]:
    """Run sustained matmul burn and return time-series metrics."""
    with cp.cuda.Device(gpu_index):
        # Allocate matrices (numpy→cupy avoids CTK requirement)
        a = cp.array(np.random.randn(size, size).astype(np.float32))
        b = cp.array(np.random.randn(size, size).astype(np.float32))

        # Start background sampler
        samples: list[dict[str, Any]] = []
        stop_event = threading.Event()
        start_time = time.perf_counter()
        sampler = threading.Thread(
            target=_sample_clocks,
            args=(gpu_index, samples, start_time, stop_event),
            daemon=True,
        )
        sampler.start()

        # Burn loop
        iterations = 0
        t_start = time.perf_counter()
        try:
            while (time.perf_counter() - t_start) < duration_sec:
                c = cp.dot(a, b)
                cp.cuda.Stream.null.synchronize()
                iterations += 1
        finally:
            stop_event.set()
            sampler.join(timeout=2)

        elapsed = time.perf_counter() - t_start

    # Compute statistics from samples
    sm_clocks = [s["sm"] for s in samples if s["sm"] is not None]
    mem_clocks = [s["mem"] for s in samples if s["mem"] is not None]
    temps = [s["temp"] for s in samples if s["temp"] is not None]
    powers = [s["power"] for s in samples if s["power"] is not None]

    sm_initial = sm_clocks[0] if sm_clocks else None
    sm_min = min(sm_clocks) if sm_clocks else None
    throttling = False
    if sm_initial and sm_min and sm_initial > 0:
        throttling = (sm_initial - sm_min) / sm_initial > 0.15

    return {
        "duration_sec": round(elapsed, 1),
        "iterations": iterations,
        "size": size,
        "sm_initial_mhz": sm_initial,
        "sm_min_mhz": sm_min,
        "sm_avg_mhz": round(sum(sm_clocks) / len(sm_clocks)) if sm_clocks else None,
        "mem_initial_mhz": mem_clocks[0] if mem_clocks else None,
        "mem_avg_mhz": round(sum(mem_clocks) / len(mem_clocks)) if mem_clocks else None,
        "temp_max_c": max(temps) if temps else None,
        "temp_avg_c": round(sum(temps) / len(temps)) if temps else None,
        "power_avg_w": round(sum(powers) / len(powers)) if powers else None,
        "throttling_detected": throttling,
        "time_series": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sustained GPU compute burn test")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index")
    parser.add_argument("--duration", type=int, default=30, help="Burn duration in seconds")
    parser.add_argument("--size", type=int, default=8192, help="Matrix size (N for NxN)")
    args = parser.parse_args()

    try:
        with cp.cuda.Device(args.gpu):
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            needed = args.size * args.size * 4 * 3  # 3 matrices × float32
            if needed > free_bytes * 0.9:
                args.size = int((free_bytes * 0.9 / (4 * 3)) ** 0.5)
                args.size = (args.size // 512) * 512
                if args.size < 512:
                    print(json.dumps({"error": "insufficient GPU memory for burn test"}))
                    sys.exit(1)

        metrics = run_burn(args.gpu, args.size, args.duration)

        props = cp.cuda.runtime.getDeviceProperties(args.gpu)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode()

        result = {
            "benchmark": "burn",
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
