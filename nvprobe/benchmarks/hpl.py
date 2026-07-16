"""HPL (High Performance Linpack) benchmark wrapper."""

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


class HplBenchmark(BaseBenchmark):
    """Wrapper around NVIDIA HPL benchmark (xhpl) — requires MPI."""

    name = "hpl"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        binary = self.params.get("binary", "xhpl")
        problem_sizes = self.params.get("problem_sizes", [2048])

        if not shutil.which(binary) and not Path(binary).is_file():
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error=f"HPL binary '{binary}' not found. Run 'nvprobe setup-tools' or install xhpl.",
            )

        mpi_run = _find_mpi_run()
        env = _build_env(gpu_index)

        try:
            if mpi_run:
                cmd = [mpi_run, "--allow-run-as-root", "-np", "1", binary,
                       "--problem-size", str(problem_sizes[0])]
            else:
                cmd = [binary, "--problem-size", str(problem_sizes[0])]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, check=True,
                env=env,
            )
            gflops = _parse_hpl_output(proc.stdout)
            return BenchmarkResult(
                benchmark=self.name,
                gpu_model="unknown",
                gpu_index=gpu_index,
                precision=precision,
                batch_size=batch_size,
                metrics={"gflops": gflops, "problem_size": problem_sizes[0]},
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
        binary = self.params.get("binary", "xhpl")
        problem_sizes = self.params.get("problem_sizes", [2048])

        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

for PS in {' '.join(str(s) for s in problem_sizes)}; do
    mpirun -np 1 {binary} --problem-size $PS
done
"""


def _parse_hpl_output(output: str) -> float:
    """Extract GFLOPS from HPL output."""
    for line in output.splitlines():
        line_lower = line.lower()
        if "gflops" in line_lower:
            parts = line.split()
            for part in parts:
                try:
                    return float(part)
                except ValueError:
                    continue
        if "hpl_outof" in line_lower or "g_flops" in line_lower:
            parts = line.split("=")
            if len(parts) > 1:
                try:
                    return float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    continue
    return 0.0
