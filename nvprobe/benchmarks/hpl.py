"""HPL (High Performance Linpack) benchmark wrapper."""

from __future__ import annotations

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


def _run_hpl_size(
    binary: str, size: int, mpi_run: str | None,
    env: dict[str, str], gpu_index: int, precision: str, batch_size: int,
) -> BenchmarkResult | None:
    """Run HPL for a single problem size. Returns None if binary not found."""
    try:
        if mpi_run:
            cmd = [mpi_run, "-np", "1", binary, "--problem-size", str(size)]
        else:
            cmd = [binary, "--problem-size", str(size)]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, check=True,
            env=env,
        )
        gflops = _parse_hpl_output(proc.stdout)
        return BenchmarkResult(
            benchmark="hpl", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            metrics={"gflops": gflops, "problem_size": size},
            raw_output=proc.stdout,
        )
    except FileNotFoundError:
        return BenchmarkResult(
            benchmark="hpl", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            success=False,
            error="MPI binary not found. Install OpenMPI or MPICH.",
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        detail = stderr.strip()[-500:] if stderr else stdout.strip()[-500:]
        return BenchmarkResult(
            benchmark="hpl", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            success=False, error=f"{exc}\n{detail}".strip(),
        )


class HplBenchmark(BaseBenchmark):
    """Wrapper around NVIDIA HPL benchmark (xhpl) — requires MPI."""

    name = "hpl"
    uses_precision_batch = False

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
        last_result = None

        for size in problem_sizes:
            result = _run_hpl_size(binary, size, mpi_run, env, gpu_index, precision, batch_size)
            if result is None or result.success:
                last_result = result or last_result
                if result and not result.success:
                    break
                continue
            # If MPI failed, retry without MPI
            if mpi_run and "opal_pmix" in result.error or "orte" in result.error:
                result2 = _run_hpl_size(binary, size, None, env, gpu_index, precision, batch_size)
                if result2:
                    last_result = result2
                    if not result2.success:
                        break
                else:
                    last_result = result
            else:
                last_result = result
                break

        return last_result or BenchmarkResult(
            benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            success=False, error="No problem sizes configured",
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
