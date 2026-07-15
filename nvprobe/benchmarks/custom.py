"""Custom CUDA kernels benchmark — matmul, conv2d, attention microbenchmarks."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult


class CustomCudaBenchmark(BaseBenchmark):
    """Benchmark user-defined CUDA kernels (matmul, conv2d, attention)."""

    name = "custom"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        kernels = self.params.get("kernels", ["matmul"])
        matrix_sizes = self.params.get("matrix_sizes", [1024])
        iterations = self.params.get("iterations", 50)

        cmd = [
            sys.executable, "-m", "nvprobe.benchmarks._cuda.custom_kernels",
            "--gpu", str(gpu_index),
            "--kernels", ",".join(kernels),
            "--sizes", ",".join(str(s) for s in matrix_sizes),
            "--iterations", str(iterations),
            "--precision", precision,
            "--batch-size", str(batch_size),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200, check=True,
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
        kernels = self.params.get("kernels", ["matmul"])
        matrix_sizes = self.params.get("matrix_sizes", [1024])
        iterations = self.params.get("iterations", 50)
        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

python3 -m nvprobe.benchmarks._cuda.custom_kernels \\
    --gpu 0 \\
    --kernels {','.join(kernels)} \\
    --sizes {','.join(str(s) for s in matrix_sizes)} \\
    --iterations {iterations} \\
    --precision {precision} \\
    --batch-size {batch_size}
"""
