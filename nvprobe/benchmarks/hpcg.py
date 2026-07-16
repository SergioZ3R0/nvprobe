"""HPCG (High Performance Conjugate Gradients) benchmark wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


def _find_mpi_run() -> str | None:
    for name in ["mpirun", "srun"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _build_env(gpu_index: int) -> dict[str, str]:
    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def _run_hpcg_size(
    binary: str, size: int, mpi_run: str | None,
    env: dict[str, str], gpu_index: int, precision: str, batch_size: int,
) -> BenchmarkResult | None:
    try:
        rt_seconds = 60
        if mpi_run:
            cmd = [mpi_run, "-np", "1", binary,
                   f"--nx={size}", f"--ny={size}", f"--nz={size}",
                   f"--rt={rt_seconds}"]
        else:
            cmd = [binary,
                   f"--nx={size}", f"--ny={size}", f"--nz={size}",
                   f"--rt={rt_seconds}"]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, check=True,
            env=env,
        )
        gflops = _parse_hpcg_output(proc.stdout)
        return BenchmarkResult(
            benchmark="hpcg", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            metrics={"gflops": gflops, "grid_size": size, "run_time": rt_seconds},
            raw_output=proc.stdout,
        )
    except FileNotFoundError:
        return BenchmarkResult(
            benchmark="hpcg", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            success=False,
            error="MPI binary not found. Install OpenMPI or MPICH.",
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        detail = stderr.strip()[-500:] if stderr else stdout.strip()[-500:]
        if "cannot open shared object file" in detail:
            if "libcublas" in detail:
                from nvprobe.benchmarks.base import _find_system_cuda_libs, _find_cupy_cuda_libs
                searched = _find_system_cuda_libs() or []
                cupy_libs = _find_cupy_cuda_libs() or []
                detail += (
                    "\n\nCUDA runtime library not found. Install CUDA toolkit:\n"
                    "  - For NVIDIA HPC Benchmarks (CUDA 12): install CUDA 12.x\n"
                    "  - Or ensure cupy-cuda12x[ctk] is installed (bundles CUDA libs)\n"
                    f"  Searched system CUDA paths: {searched}\n"
                    f"  Searched cupy paths: {cupy_libs}\n"
                    "  Check: module avail cuda, module load cuda/12.x, or set CUDA_HOME"
                )
            elif "libmpi" in detail:
                from nvprobe.benchmarks.base import subprocess_env
                mpi_envs = ["MPI_HOME", "OPAL_PREFIX", "I_MPI_ROOT"]
                mpi_vals = {v: os.environ.get(v, "(not set)") for v in mpi_envs}
                detail += (
                    "\n\nMPI library not found. Install an MPI implementation or load a module:\n"
                    "  - Ubuntu/Debian: sudo apt install mpich\n"
                    "  - RHEL/CentOS: sudo dnf install mpich\n"
                    "  - Cluster: module avail mpi, module load mpi/openmpi\n"
                    f"  MPI env vars: {mpi_vals}\n"
                    "  Check: which mpirun, mpirun --version"
                )
        return BenchmarkResult(
            benchmark="hpcg", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            success=False, error=f"{exc}\n{detail}".strip(),
        )


class HpcgBenchmark(BaseBenchmark):
    """Wrapper around NVIDIA HPCG benchmark (xhpcg) — requires MPI."""

    name = "hpcg"
    uses_precision_batch = False
    size_keys = ["grid_sizes"]

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        binary = self.params.get("binary", "xhpcg")
        binary_path = Path(binary).expanduser()
        grid_sizes = self.params.get("grid_sizes", [128])

        if not shutil.which(str(binary_path)) and not binary_path.is_file():
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error=f"HPCG binary '{binary}' not found. Run 'nvprobe setup-tools' or install xhpcg.",
            )

        mpi_run = _find_mpi_run()
        env = _build_env(gpu_index)
        last_result = None
        binary_str = str(binary_path)

        for size in grid_sizes:
            result = _run_hpcg_size(binary_str, size, mpi_run, env, gpu_index, precision, batch_size)
            if result is None or result.success:
                last_result = result or last_result
                if result and not result.success:
                    break
                continue
            if mpi_run and ("opal_pmix" in result.error or "orte" in result.error):
                result2 = _run_hpcg_size(binary_str, size, None, env, gpu_index, precision, batch_size)
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
            success=False, error="No grid sizes configured",
        )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        binary = self.params.get("binary", "xhpcg")
        binary_path = Path(binary).expanduser()
        grid_sizes = self.params.get("grid_sizes", [128])

        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

for GS in {' '.join(str(s) for s in grid_sizes)}; do
    mpirun -np 1 {binary_path} --nx=$GS --ny=$GS --nz=$GS --rt=60
done
"""


def _parse_hpcg_output(output: str) -> float:
    for line in output.splitlines():
        lower = line.lower()
        if "gflop" in lower:  # matches "gflops", "GFLOP/s", "GFLOPS"
            parts = line.split()
            for part in parts:
                try:
                    return float(part)
                except ValueError:
                    continue
    return 0.0
