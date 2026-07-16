"""Base class for all benchmark modules."""

from __future__ import annotations

import glob
import os
import shutil
import site
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _find_cupy_cuda_libs() -> list[str]:
    """Find CuPy's bundled CUDA library directories (from [ctk] extras)."""
    dirs: list[str] = []
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"]
            if name.startswith("cupy-cuda"):
                # Use public locate_file() API (Python 3.9+), fall back to _path
                try:
                    loc = dist.locate_file("").parent  # site-packages/
                except Exception:
                    loc = dist._path.parent
                # [ctk] installs libs under cupy_cudaXXX.libs/
                for pattern in glob.glob(str(loc / f"{name.replace('-', '_')}.libs" / "*")):
                    d = os.path.dirname(pattern)
                    if d not in dirs:
                        dirs.append(d)
                # Also check the package dir itself
                pkg_dir = str(loc / name.replace("-", "_"))
                for sub in ("lib", "lib64", ".libs"):
                    p = os.path.join(pkg_dir, sub)
                    if os.path.isdir(p):
                        dirs.append(p)
    except Exception:
        pass
    # Fallback: search common site-packages for any cupy_cuda*.libs
    if not dirs:
        candidates = set()
        try:
            candidates.add(site.getusersitepackages())
        except Exception:
            pass
        try:
            candidates.update(site.getsitepackages())
        except Exception:
            pass
        for sp in candidates:
            if sp and os.path.isdir(sp):
                for d in glob.glob(os.path.join(sp, "cupy_cuda*.libs")):
                    if os.path.isdir(d) and d not in dirs:
                        dirs.append(d)
    return dirs


def _find_system_cuda_libs() -> list[str]:
    """Find system CUDA library directories."""
    candidates = [
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
    ]
    # Check CUDA_PATH/CUDA_HOME/CUDA_ROOT env vars (common on HPC)
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            candidates.insert(0, os.path.join(cuda_path, "lib64"))
            candidates.insert(0, os.path.join(cuda_path, "compat"))
    # Find nvcc via shutil.which and derive toolkit lib path
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit_base = str(Path(nvcc).parent.parent)
        for sub in ("lib64", "lib", "compat"):
            p = os.path.join(toolkit_base, sub)
            if p not in candidates:
                candidates.insert(0, p)
    # Find nvidia-smi and check nearby lib dirs
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        nvidia_dir = str(Path(nvidia_smi).parent)
        for sub in ("", "../lib64", "../lib", "../compat"):
            p = os.path.normpath(os.path.join(nvidia_dir, sub))
            if p not in candidates:
                candidates.insert(0, p)

    found: list[str] = []
    for path in candidates:
        p = os.path.realpath(path)
        if not os.path.isdir(p):
            continue
        try:
            files = os.listdir(p)
        except OSError:
            continue
        prefixes = ("libcuda.so", "libcudart.so", "libcublas.so", "libcudnn.so")
        if any(any(f.startswith(prefix) for f in files) for prefix in prefixes):
            found.append(p)
    return found


def _find_nccl_libs() -> list[str]:
    """Find NCCL library directories (system or pip-installed nvidia-nccl-cuXX)."""
    dirs: list[str] = []

    # 1. Check pip-installed nvidia-nccl-cuXX package
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"] or ""
            if name.startswith("nvidia-nccl-cu"):
                try:
                    loc = dist.locate_file("").parent  # site-packages/
                except Exception:
                    loc = dist._path.parent
                nccl_lib = str(loc / "nvidia" / "nccl" / "lib")
                if os.path.isdir(nccl_lib) and nccl_lib not in dirs:
                    dirs.append(nccl_lib)
    except Exception:
        pass

    # 2. Search system paths for libnccl.so
    candidates = [
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
    ]
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            candidates.insert(0, os.path.join(cuda_path, "lib64"))
            candidates.insert(0, os.path.join(cuda_path, "compat"))
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit_base = str(Path(nvcc).parent.parent)
        for sub in ("lib64", "lib", "compat"):
            p = os.path.join(toolkit_base, sub)
            if p not in candidates:
                candidates.insert(0, p)
    # Also check paths derived from nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        nvidia_dir = str(Path(nvidia_smi).parent)
        for sub in ("", "../lib64", "../lib", "../compat"):
            p = os.path.normpath(os.path.join(nvidia_dir, sub))
            if p not in candidates:
                candidates.insert(0, p)

    for path in candidates:
        p = os.path.realpath(path)
        if not os.path.isdir(p):
            continue
        try:
            files = os.listdir(p)
        except OSError:
            continue
        if any(f.startswith("libnccl.so") for f in files):
            if p not in dirs:
                dirs.append(p)

    return dirs


# Libraries that NVIDIA HPC benchmark binaries may require at runtime.
KNOWN_MISSING_LIBS = ("libcublas", "libmpi", "libnccl")


def _guess_cuda_major() -> str:
    """Roughly detect CUDA major version for diagnostic messages."""
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.splitlines():
            if "release" in line:
                ver = line.split("release")[-1].strip().rstrip(",").split(",")[0]
                return ver.split(".")[0]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        ver = out.stdout.strip()
        if ver:
            major = ver.split(".")[0]
            return "12" if int(major) >= 535 else "11"
    except Exception:
        pass
    return "12"


def _diagnose_missing_lib(lib_name: str, detail: str) -> str:
    """Augment *detail* with a diagnostic hint for a missing shared library.

    *lib_name* is one of the entries in ``KNOWN_MISSING_LIBS`` (e.g.
    ``"libcublas"``, ``"libmpi"``, ``"libnccl"``).
    Returns the original *detail* with an appended diagnostic message.
    """
    if lib_name == "libcublas":
        searched = _find_system_cuda_libs() or []
        cupy_libs = _find_cupy_cuda_libs() or []
        detail += (
            "\n\nCUDA runtime library not found. Install CUDA toolkit or a compatible runtime:\n"
            "  - For NVIDIA HPC Benchmarks (CUDA 12): install CUDA 12.x\n"
            "  - Or ensure cupy-cuda12x[ctk] is installed (bundles CUDA libs)\n"
            f"  Searched system CUDA paths: {searched}\n"
            f"  Searched cupy paths: {cupy_libs}\n"
            "  Check: module avail cuda, module load cuda/12.x, or set CUDA_HOME"
        )
    elif lib_name == "libmpi":
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
    elif lib_name == "libnccl":
        nccl_system = _find_nccl_libs()
        cupy_libs = _find_cupy_cuda_libs() or []
        cuda_ver = _guess_cuda_major()
        detail += (
            "\n\nNCCL library not found. The NVIDIA HPC Benchmarks binaries require "
            "libnccl.so at runtime.\n"
            "  Install via pip (recommended):\n"
            f"    pip install --user nvidia-nccl-cu{cuda_ver}\n"
            "  Or install a system NCCL package:\n"
            "    Ubuntu/Debian: sudo apt install libnccl2 libnccl-dev\n"
            "    RHEL/CentOS:   sudo dnf install nccl\n"
            "  Or load a NCCL module on HPC clusters:\n"
            "    module avail nccl, module load nccl\n"
            f"  Searched pip nvidia-nccl-cu* paths: {nccl_system}\n"
            f"  Searched cupy paths: {cupy_libs}\n"
            "  Check: CUDA_HOME/lib64, /usr/lib/x86_64-linux-gnu/libnccl*"
        )
    return detail


def subprocess_env() -> dict[str, str]:
    """Return an environment dict with user site-packages and CUDA library paths."""
    env = os.environ.copy()

    # Add user site-packages to PYTHONPATH
    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(user_site):
        pythonpath = env.get("PYTHONPATH", "")
        parts = [p for p in pythonpath.split(os.pathsep) if p]
        if user_site not in parts:
            parts.insert(0, user_site)
        env["PYTHONPATH"] = os.pathsep.join(parts)

    # Add user local bin and CUDA bin to PATH (for mlcr, nvcc, etc.)
    user_local_bin = os.path.join(str(Path.home()), ".local", "bin")
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    for extra in [user_local_bin]:
        if extra not in path_parts:
            path_parts.insert(0, extra)
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = env.get(env_var)
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if cuda_bin not in path_parts:
                path_parts.insert(0, cuda_bin)
    env["PATH"] = os.pathsep.join(path_parts)

    # Add CUDA library paths to LD_LIBRARY_PATH
    cuda_libs: list[str] = []
    cuda_libs.extend(_find_cupy_cuda_libs())
    cuda_libs.extend(_find_system_cuda_libs())
    cuda_libs.extend(_find_nccl_libs())

    # Add MPI lib paths (for HPL/HPCG binaries compiled against MPI).
    # Only derive paths from mpirun (NOT srun — srun is the Slurm launcher,
    # its location is unrelated to MPI libraries).
    for mpi_name in ["mpirun"]:
        mpi_bin = shutil.which(mpi_name)
        if mpi_bin:
            mpi_base = str(Path(mpi_bin).parent.parent)
            for sub in ("lib", "lib64"):
                ml = os.path.join(mpi_base, sub)
                if os.path.isdir(ml) and ml not in cuda_libs:
                    cuda_libs.append(ml)
    # Also check common MPI environment variables set by module systems
    for env_var in ("MPI_HOME", "OPAL_PREFIX", "I_MPI_ROOT"):
        mpi_path = os.environ.get(env_var)
        if mpi_path:
            for sub in ("lib", "lib64"):
                ml = os.path.join(mpi_path, sub)
                if os.path.isdir(ml) and ml not in cuda_libs:
                    cuda_libs.append(ml)

    if cuda_libs:
        existing = env.get("LD_LIBRARY_PATH", "")
        existing_parts = [p for p in existing.split(os.pathsep) if p]
        for lib in cuda_libs:
            if lib not in existing_parts:
                existing_parts.insert(0, lib)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(existing_parts)

    return env


@dataclass
class BenchmarkResult:
    """Result from a single benchmark execution."""

    benchmark: str
    gpu_model: str
    gpu_index: int
    precision: str
    batch_size: int
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    success: bool = True
    error: str = ""


class BaseBenchmark(ABC):
    """Abstract base for all benchmark implementations."""

    name: str = "base"
    uses_precision_batch: bool = True  # False for HPC benchmarks that ignore these

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @abstractmethod
    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        """Run the benchmark locally on a specific GPU. Returns result."""

    @abstractmethod
    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return sbatch script content for this benchmark."""

    def parse_slurm_output(self, output: str, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        """Parse Slurm job output into a BenchmarkResult. Override for custom parsing."""
        return BenchmarkResult(
            benchmark=self.name,
            gpu_model="unknown",
            gpu_index=gpu_index,
            precision=precision,
            batch_size=batch_size,
            raw_output=output,
        )
