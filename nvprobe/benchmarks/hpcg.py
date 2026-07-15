"""HPCG (High Performance Conjugate Gradients) benchmark wrapper."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult


class HpcgBenchmark(BaseBenchmark):
    """Wrapper around a pre-compiled HPCG binary (xhpcg) via Slurm."""

    name = "hpcg"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        binary = self.params.get("binary", "xhpcg")
        grid_sizes = self.params.get("grid_sizes", [128])

        try:
            proc = subprocess.run(
                [binary, "--grid", str(grid_sizes[0])],
                capture_output=True, text=True, timeout=3600, check=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)},
            )
            gflops = _parse_hpcg_output(proc.stdout)
            return BenchmarkResult(
                benchmark=self.name,
                gpu_model="unknown",
                gpu_index=gpu_index,
                precision=precision,
                batch_size=batch_size,
                metrics={"gflops": gflops, "grid_size": grid_sizes[0]},
                raw_output=proc.stdout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False, error=str(exc),
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return shell commands for this benchmark (without SBATCH headers)."""
        binary = self.params.get("binary", "xhpcg")
        grid_sizes = self.params.get("grid_sizes", [128])
        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

for GS in {' '.join(str(s) for s in grid_sizes)}; do
    {binary} --grid $GS
done
"""


def _parse_hpcg_output(output: str) -> float:
    """Extract GFLOPS from HPCG output."""
    for line in output.splitlines():
        if "gflops" in line.lower():
            parts = line.split()
            for part in parts:
                try:
                    return float(part)
                except ValueError:
                    continue
    return 0.0
