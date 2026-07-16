"""Base class for all benchmark modules."""

from __future__ import annotations

import glob
import os
import shutil
import site
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
                loc = dist._path.parent  # site-packages/
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

    # Also add MPI lib paths (for HPL/HPCG binaries compiled against MPI)
    for mpi_name in ["mpirun", "srun"]:
        mpi_bin = shutil.which(mpi_name)
        if mpi_bin:
            mpi_base = str(Path(mpi_bin).parent.parent)
            for sub in ("lib", "lib64"):
                ml = os.path.join(mpi_base, sub)
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
