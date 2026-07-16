"""Base class for all benchmark modules."""

from __future__ import annotations

import glob
import os
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
    nvidia_smi = os.popen("which nvidia-smi 2>/dev/null").read().strip()
    if nvidia_smi:
        nvidia_dir = str(Path(nvidia_smi).parent)
        candidates.insert(0, nvidia_dir)
        candidates.insert(0, os.path.join(nvidia_dir, "..", "compat"))
        candidates.insert(0, os.path.join(nvidia_dir, "..", "lib64"))

    found: list[str] = []
    for path in candidates:
        p = os.path.realpath(path)
        if os.path.isdir(p) and any(
            os.path.exists(os.path.join(p, f"libcuda.so.{v}")) for v in ("1", "")
        ):
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

    # Add CUDA library paths to LD_LIBRARY_PATH
    cuda_libs: list[str] = []
    cuda_libs.extend(_find_cupy_cuda_libs())
    cuda_libs.extend(_find_system_cuda_libs())

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
