"""GPU memory test benchmark — detects VRAM faults via multiple test patterns."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


class MemtestBenchmark(BaseBenchmark):
    """GPU VRAM integrity test using solid, checkerboard, random, and walking-1 patterns."""

    name = "memtest"
    uses_precision_batch = False

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        sizes_mb = self.params.get("sizes_mb", [1024])
        for size_mb in sizes_mb:
            cmd = [
                sys.executable, "-m", "nvprobe.benchmarks._cuda.memtest",
                "--gpu", str(gpu_index),
                "--size", str(size_mb),
            ]

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, check=True,
                    env=subprocess_env(),
                )
                data = json.loads(proc.stdout)
                if "error" in data:
                    return BenchmarkResult(
                        benchmark=self.name, gpu_model="unknown",
                        gpu_index=gpu_index, precision=precision, batch_size=batch_size,
                        success=False, error=data["error"],
                    )
                return BenchmarkResult(
                    benchmark=self.name,
                    gpu_model=data.get("gpu_model", "unknown"),
                    gpu_index=gpu_index,
                    precision=precision,
                    batch_size=batch_size,
                    metrics=data.get("metrics", {}),
                    raw_output=proc.stdout,
                )
            except (subprocess.CalledProcessError, json.JSONDecodeError,
                    subprocess.TimeoutExpired) as exc:
                stderr = getattr(exc, "stderr", "") or ""
                stdout = getattr(exc, "stdout", "") or ""
                error_msg = f"{exc}"
                if stdout:
                    try:
                        data = json.loads(stdout)
                        if "error" in data:
                            error_msg = data["error"]
                    except json.JSONDecodeError:
                        error_msg += f"\n{stdout.strip()}"
                if stderr:
                    error_msg += f"\n{stderr.strip()}"
                return BenchmarkResult(
                    benchmark=self.name, gpu_model="unknown",
                    gpu_index=gpu_index, precision=precision, batch_size=batch_size,
                    success=False, error=error_msg,
                )

        return BenchmarkResult(
            benchmark=self.name, gpu_model="unknown",
            gpu_index=gpu_index, precision=precision, batch_size=batch_size,
            success=False, error="no valid sizes configured",
        )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        sizes_mb = self.params.get("sizes_mb", [1024])
        size = sizes_mb[0] if sizes_mb else 1024
        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

python3 -m nvprobe.benchmarks._cuda.memtest \\
    --gpu 0 \\
    --size {size}
"""
