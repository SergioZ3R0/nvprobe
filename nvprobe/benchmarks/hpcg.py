"""HPCG (High Performance Conjugate Gradients) benchmark wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


def _find_mpi_run() -> str | None:
    """Find mpirun or srun."""
    for name in ["mpirun", "srun"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _build_env(gpu_index: int) -> dict[str, str]:
    """Build environment with CUDA libraries and GPU selection."""
    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


class HpcgBenchmark(BaseBenchmark):
    """Wrapper around NVIDIA HPCG benchmark (xhpcg) — requires MPI."""

    name = "hpcg"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        binary = self.params.get("binary", "xhpcg")
        grid_sizes = self.params.get("grid_sizes", [128])

        if not shutil.which(binary) and not Path(binary).is_file():
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error=f"HPCG binary '{binary}' not found. Run 'nvprobe setup-tools' or install xhpcg.",
            )

        mpi_run = _find_mpi_run()
        env = _build_env(gpu_index)

        try:
            if mpi_run:
                cmd = [mpi_run, "--allow-run-as-root", "-np", "1", binary,
                       "--grid", str(grid_sizes[0])]
            else:
                cmd = [binary, "--grid", str(grid_sizes[0])]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, check=True,
                env=env,
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
        except FileNotFoundError:
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error="MPI not found. Install OpenMPI or MPICH: apt install libopenmpi-dev / yum install openmpi-devel",
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
    mpirun -np 1 {binary} --grid $GS
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
