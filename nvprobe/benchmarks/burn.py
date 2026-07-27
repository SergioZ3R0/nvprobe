"""Sustained GPU compute burn benchmark — detects thermal/power throttling over time."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


class BurnBenchmark(BaseBenchmark):
    """Sustained matmul burn to detect clock degradation from thermal/power throttling."""

    name = "burn"
    uses_precision_batch = False

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        duration = self.params.get("duration_sec", 30)
        size = self.params.get("matrix_size", 8192)

        cmd = [
            sys.executable, "-m", "nvprobe.benchmarks._cuda.burn",
            "--gpu", str(gpu_index),
            "--duration", str(duration),
            "--size", str(size),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=duration + 30, check=True,
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
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown",
                gpu_index=gpu_index, precision=precision, batch_size=batch_size,
                success=False, error=f"{exc}\n{stderr}".strip(),
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        duration = self.params.get("duration_sec", 30)
        size = self.params.get("matrix_size", 8192)
        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

python3 -m nvprobe.benchmarks._cuda.burn \\
    --gpu 0 \\
    --duration {duration} \\
    --size {size}
"""
