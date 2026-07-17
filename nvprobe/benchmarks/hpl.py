"""HPL (High Performance Linpack) benchmark wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from nvprobe.benchmarks.base import (
    BaseBenchmark, BenchmarkResult, KNOWN_MISSING_LIBS,
    _diagnose_missing_lib, subprocess_env,
)


def _find_mpi_run() -> str | None:
    for name in ["mpirun", "srun"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _get_gpu_memory_mb(gpu_index: int) -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
             "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        pass
    return None


def _calculate_hpl_n(gpu_index: int) -> int:
    mem_mb = _get_gpu_memory_mb(gpu_index)
    if mem_mb is None:
        return 4096
    mem_bytes = mem_mb * 1024 * 1024
    n = int((mem_bytes * 0.8 / 8) ** 0.5)  # FP64: 8 bytes per element, 80% utilization
    n = (n // 64) * 64  # align to 64
    return max(1024, min(n, 262144))


def _generate_hpl_dat(n: int, nb: int = 1024, p: int = 1, q: int = 1) -> str:
    return (
        f"HPLinpack benchmark input file\n"
        f"Innovative Computing Laboratory, University of Tennessee\n"
        f"HPL.out      output file name (if any)\n"
        f"6            device out (6=stdout,7=stderr,file)\n"
        f"1            # of problems sizes (N)\n"
        f"{n}          Ns\n"
        f"1            # of NBs\n"
        f"{nb}         NBs\n"
        f"0            PMAP process mapping (0=Row-,1=Column-major)\n"
        f"1            # of process grids (P x Q)\n"
        f"{p}          Ps\n"
        f"{q}          Qs\n"
        f"16.0         threshold\n"
        f"1            # of panel fact\n"
        f"2            PFACTs (0=left, 1=Crout, 2=Right)\n"
        f"1            # of recursive stopping criterias\n"
        f"4            NBMINs (>= 1)\n"
        f"1            # of panels in recursion\n"
        f"2            NDIVs\n"
        f"1            # of recursive panel fact.\n"
        f"1            RFACTs (0=left, 1=Crout, 2=Right)\n"
        f"1            # of broadcast\n"
        f"1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)\n"
        f"1            # of lookahead depth\n"
        f"0            DEPTHs (>=0)\n"
        f"2            SWAP (0=bin-exch,1=long,2=mix)\n"
        f"64           swapping threshold\n"
        f"0            L1 in (0=transposed,1=no-transposed) form\n"
        f"0            U  in (0=transposed,1=no-transposed) form\n"
        f"1            Equilibration (0=no,1=yes)\n"
        f"8            memory alignment in (>0)\n"
    )


def _build_env(gpu_index: int) -> dict[str, str]:
    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def _run_hpl_size(
    binary: str, n: int, mpi_run: str | None,
    env: dict[str, str], gpu_index: int, precision: str, batch_size: int,
) -> BenchmarkResult | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            hpl_dat_path = os.path.join(tmpdir, "HPL.dat")
            with open(hpl_dat_path, "w") as f:
                f.write(_generate_hpl_dat(n))

            if mpi_run:
                cmd = [mpi_run, "-np", "1", binary]
            else:
                cmd = [binary]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, check=True,
                env=env, cwd=tmpdir,
            )
            gflops = _parse_hpl_output(proc.stdout)
            return BenchmarkResult(
                benchmark="hpl", gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                metrics={"gflops": gflops, "problem_size": n},
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
        if "cannot open shared object file" in detail:
            for lib in KNOWN_MISSING_LIBS:
                if lib in detail:
                    detail = _diagnose_missing_lib(lib, detail)
                    break
        return BenchmarkResult(
            benchmark="hpl", gpu_model="unknown", gpu_index=gpu_index,
            precision=precision, batch_size=batch_size,
            success=False, error=f"{exc}\n{detail}".strip(),
        )


class HplBenchmark(BaseBenchmark):
    """Wrapper around NVIDIA HPL benchmark (xhpl) — requires MPI."""

    name = "hpl"
    uses_precision_batch = False
    size_keys = ["problem_sizes"]

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        binary = self.params.get("binary", "xhpl")
        binary_path = Path(binary).expanduser()
        problem_sizes = self.params.get("problem_sizes", [])

        if not problem_sizes:
            n = _calculate_hpl_n(gpu_index)
            problem_sizes = [n]

        if not shutil.which(str(binary_path)) and not binary_path.is_file():
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error=f"HPL binary '{binary}' not found. Run 'nvprobe setup-tools' or install xhpl.",
            )

        mpi_run = _find_mpi_run()
        env = _build_env(gpu_index)
        last_result = None
        binary_str = str(binary_path)

        for n in problem_sizes:
            result = _run_hpl_size(binary_str, n, mpi_run, env, gpu_index, precision, batch_size)
            if result is None or result.success:
                last_result = result or last_result
                if result and not result.success:
                    break
                continue
            if mpi_run and ("opal_pmix" in result.error or "orte" in result.error):
                result2 = _run_hpl_size(binary_str, n, None, env, gpu_index, precision, batch_size)
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
            success=False, error="No problem sizes configured",
        )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        binary = self.params.get("binary", "xhpl")
        binary_path = Path(binary).expanduser()
        problem_sizes = self.params.get("problem_sizes", [])
        if not problem_sizes:
            n = _calculate_hpl_n(gpu_index)
            problem_sizes = [n]

        lines = [f"export CUDA_VISIBLE_DEVICES={gpu_index}", ""]
        for n in problem_sizes:
            hpl_dat = _generate_hpl_dat(n)
            lines.append("cat > HPL.dat << 'EOF'")
            lines.append(hpl_dat.strip())
            lines.append("EOF")
            lines.append(f"mpirun -np 1 {binary_path}")
            lines.append("")
        return "\n".join(lines)


def _parse_hpl_output(output: str) -> float:
    lines = output.splitlines()
    best = 0.0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("---"):
            if i + 1 < len(lines):
                parts = lines[i + 1].split()
                if len(parts) >= 7:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        continue
        if stripped.startswith("WC") or stripped.startswith("Wc"):
            parts = stripped.split()
            if len(parts) >= 7:
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
        if "gflops" in stripped.lower():
            for token in stripped.split():
                try:
                    return float(token)
                except ValueError:
                    continue
    return best
