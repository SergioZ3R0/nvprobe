"""HPCG (High Performance Conjugate Gradients) benchmark wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from nvprobe.benchmarks.base import (
    BaseBenchmark, BenchmarkResult, KNOWN_MISSING_LIBS,
    _diagnose_missing_lib, subprocess_env,
)

# Heuristic: bytes per grid point for HPCG GPU memory (CSR matrix + vectors + workspace).
# NVIDIA's GPU-accelerated HPCG uses ~400-500 B/point for main arrays; 1200 is very conservative.
_HPCG_GPU_BYTES_PER_POINT = 1200

# Heuristic: bytes per grid point for HPCG host memory (pinned buffers, MPI, CUDA driver).
# The host-side footprint is smaller but can trigger cgroup OOM-kill.
_HPCG_HOST_BYTES_PER_POINT = 500


def _get_available_host_memory() -> int | None:
    """Return available host memory in bytes, or None if unknown.

    Tries (in order): cgroup memory limit, psutil, /proc/meminfo.
    On Slurm clusters, the cgroup limit (if set) is the binding constraint.
    """
    # 1. Try cgroup v1 memory limit
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if "memory" in line and ":" in line:
                    cgroup_path = line.split(":")[2].strip()
                    if cgroup_path:
                        mem_limit = f"/sys/fs/cgroup/memory{cgroup_path}/memory.limit_in_bytes"
                        if os.path.exists(mem_limit):
                            with open(mem_limit) as mf:
                                val = int(mf.read().strip())
                                if val > 0 and val < 2**63:
                                    return val
    except Exception:
        pass
    # 2. Try cgroup v2 memory limit
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if "0::" in line:
                    cgroup_path = line.split("::")[1].strip()
                    if cgroup_path:
                        mem_max = f"/sys/fs/cgroup{cgroup_path}/memory.max"
                        if os.path.exists(mem_max):
                            with open(mem_max) as mf:
                                val = mf.read().strip()
                                if val and val != "max":
                                    return int(val) * 1024  # memory.max is in bytes
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _get_available_gpu_memory(gpu_index: int) -> int | None:
    """Return available GPU memory in bytes for the given device, or None if unknown."""
    try:
        import cupy as cp  # fmt: skip
        cp.cuda.Device(gpu_index).use()
        free, _ = cp.cuda.runtime.memGetInfo()
        return free
    except Exception:
        pass
    return None


def _estimate_hpcg_memory(grid_size: int) -> tuple[int, int]:
    """Return (gpu_bytes, host_bytes) estimate for a given HPCG grid size.

    The estimate uses conservative per-point heuristics and a 1.5× safety
    margin is applied by the caller.
    """
    n = grid_size ** 3
    return n * _HPCG_GPU_BYTES_PER_POINT, n * _HPCG_HOST_BYTES_PER_POINT


def _check_hpcg_memory(grid_size: int, gpu_index: int) -> tuple[bool, str]:
    """Check if there is enough memory to run HPCG with the given *grid_size*.

    Returns ``(ok, reason)``.  When *ok* is ``False``, *reason* explains
    which resource is insufficient.
    """
    gpu_need, host_need = _estimate_hpcg_memory(grid_size)
    gpu_need_mb = gpu_need / 1024 ** 2
    host_need_mb = host_need / 1024 ** 2

    # GPU memory check (best-effort, only if cupy is available)
    free_gpu = _get_available_gpu_memory(gpu_index)
    if free_gpu is not None and free_gpu < gpu_need * 1.5:
        free_gpu_mb = free_gpu / 1024 ** 2
        return False, (
            f"grid_size={grid_size} requires ~{gpu_need_mb:.0f} MB of GPU memory, "
            f"only {free_gpu_mb:.0f} MB free — skipping to avoid OOM"
        )

    # Host memory check (best-effort)
    free_host = _get_available_host_memory()
    if free_host is not None and free_host < host_need * 1.5:
        free_host_mb = free_host / 1024 ** 2
        return False, (
            f"grid_size={grid_size} requires ~{host_need_mb:.0f} MB of host RAM, "
            f"only {free_host_mb:.0f} MB available — skipping to avoid OOM-kill"
        )

    return True, ""


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

        # Exit code 137 = SIGKILL, almost always OOM
        if isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 137:
            return BenchmarkResult(
                benchmark="hpcg", gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error=f"possible OOM (process killed with SIGKILL, exit code 137)\n"
                      f"grid_size={size} exhausted available memory.\n"
                      f"This may be caused by a Slurm cgroup memory limit. "
                      f"Try a smaller grid_size or request more memory with --mem.\n"
                      f"{detail}",
            )

        if "cannot open shared object file" in detail:
            for lib in KNOWN_MISSING_LIBS:
                if lib in detail:
                    detail = _diagnose_missing_lib(lib, detail)
                    break
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
            ok, reason = _check_hpcg_memory(size, gpu_index)
            if not ok:
                if last_result is None:
                    last_result = BenchmarkResult(
                        benchmark=self.name, gpu_model="unknown",
                        gpu_index=gpu_index, precision=precision,
                        batch_size=batch_size, success=False,
                        error=f"all grid sizes skipped — {reason}",
                    )
                continue

            result = _run_hpcg_size(binary_str, size, mpi_run, env, gpu_index, precision, batch_size)
            if result is None or result.success:
                last_result = result or last_result
                if result and not result.success:
                    break
                continue
            if mpi_run and ("opal_pmix" in result.error or "orte" in result.error):
                result2 = _run_hpcg_size(binary_str, size, None, env, gpu_index, precision, batch_size)
                if result2:
                    if not result2.success and not result.success:
                        result2 = BenchmarkResult(
                            benchmark=self.name, gpu_model=result2.gpu_model,
                            gpu_index=gpu_index, precision=precision, batch_size=batch_size,
                            success=False,
                            error="attempt with mpirun:\n" + result.error
                                  + "\n\nattempt singleton:\n" + result2.error,
                        )
                    last_result = result2
                    if not last_result.success:
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
