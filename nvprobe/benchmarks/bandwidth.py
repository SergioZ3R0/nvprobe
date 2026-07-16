"""Bandwidth benchmark — measures memory bandwidth (host↔device, device↔device)."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


class BandwidthBenchmark(BaseBenchmark):
    """Memory bandwidth benchmark using cuda-memcheck or custom CUDA kernel."""

    name = "bandwidth"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        sizes_mb = self.params.get("sizes_mb", [1, 4, 16, 64, 256, 1024])
        iterations = self.params.get("iterations", 100)

        cmd = [
            sys.executable, "-m", "nvprobe.benchmarks._cuda.bandwidth_test",
            "--gpu", str(gpu_index),
            "--sizes", ",".join(str(s) for s in sizes_mb),
            "--iterations", str(iterations),
            "--precision", precision,
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, check=True,
                env=subprocess_env(),
            )
            data = json.loads(proc.stdout)
            return BenchmarkResult(
                benchmark=self.name,
                gpu_model=data.get("gpu_model", "unknown"),
                gpu_index=gpu_index,
                precision=precision,
                batch_size=batch_size,
                metrics=data.get("metrics", {}),
                raw_output=proc.stdout,
            )
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False, error=f"{exc}\n{stderr}".strip(),
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return shell commands for this benchmark (without SBATCH headers)."""
        sizes_mb = self.params.get("sizes_mb", [1, 4, 16, 64, 256, 1024])
        iterations = self.params.get("iterations", 100)
        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

python3 -m nvprobe.benchmarks._cuda.bandwidth_test \\
    --gpu 0 \\
    --sizes {','.join(str(s) for s in sizes_mb)} \\
    --iterations {iterations} \\
    --precision {precision}
"""
