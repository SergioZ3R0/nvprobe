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
                    loc = dist.locate_file("")  # site-packages/
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


def _find_nvidia_pip_lib(pip_prefix: str, subpath: str) -> list[str]:
    """Find library directories for a pip-installed NVIDIA package.

    *pip_prefix* is e.g. ``"nvidia-nccl-cu"``, ``"nvidia-cudnn-cu"``,
    ``"nvidia-nvshmem-cu"``.
    *subpath* is the relative path inside the package that contains
    ``.so`` files, e.g. ``"nvidia/nccl/lib"``.
    """
    dirs: list[str] = []
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"] or ""
            if name.startswith(pip_prefix):
                try:
                    loc = dist.locate_file("")  # site-packages/
                except Exception:
                    loc = dist._path.parent
                lib_dir = str(loc / subpath)
                if os.path.isdir(lib_dir) and lib_dir not in dirs:
                    dirs.append(lib_dir)
    except Exception:
        pass
    return dirs


def _find_nvidia_pip_root(pip_prefix: str, subpath: str) -> str | None:
    """Return the root directory of a pip-installed NVIDIA package.

    Returns the *subpath* directory only if it has a ``lib/`` child.
    Used e.g. to find the ``nvidia/cudnn`` root for ``CUDNN_ROOT``.
    """
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"] or ""
            if name.startswith(pip_prefix):
                try:
                    loc = dist.locate_file("")
                except Exception:
                    loc = dist._path.parent
                root = str(loc / subpath)
                if os.path.isdir(os.path.join(root, "lib")):
                    return root
    except Exception:
        pass
    return None


def _find_nccl_libs() -> list[str]:
    """Find NCCL library directories (system or pip-installed nvidia-nccl-cuXX)."""
    dirs = _find_nvidia_pip_lib("nvidia-nccl-cu", "nvidia/nccl/lib")

    candidates = [
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
    ]
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            p = os.path.join(cuda_path, "lib64")
            if p not in candidates:
                candidates.insert(0, p)
            p = os.path.join(cuda_path, "compat")
            if p not in candidates:
                candidates.insert(0, p)
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit_base = str(Path(nvcc).parent.parent)
        for sub in ("lib64", "lib", "compat"):
            p = os.path.join(toolkit_base, sub)
            if p not in candidates:
                candidates.insert(0, p)
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


def _find_cudnn_libs() -> list[str]:
    """Find cuDNN library directories (pip-installed nvidia-cudnn-cuXX or system)."""
    dirs = _find_nvidia_pip_lib("nvidia-cudnn-cu", "nvidia/cudnn/lib")

    candidates = [
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
    ]
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            p = os.path.join(cuda_path, "lib64")
            if p not in candidates:
                candidates.insert(0, p)
            p = os.path.join(cuda_path, "compat")
            if p not in candidates:
                candidates.insert(0, p)
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit_base = str(Path(nvcc).parent.parent)
        for sub in ("lib64", "lib", "compat"):
            p = os.path.join(toolkit_base, sub)
            if p not in candidates:
                candidates.insert(0, p)
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
        if any(f.startswith("libcudnn.so") for f in files):
            if p not in dirs:
                dirs.append(p)

    return dirs


def _find_nvshmem_libs() -> list[str]:
    """Find NVSHMEM library directories (pip-installed nvidia-nvshmem-cuXX or system)."""
    dirs = _find_nvidia_pip_lib("nvidia-nvshmem-cu", "nvidia/nvshmem/lib")

    candidates = [
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
    ]
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            p = os.path.join(cuda_path, "lib64")
            if p not in candidates:
                candidates.insert(0, p)
            p = os.path.join(cuda_path, "compat")
            if p not in candidates:
                candidates.insert(0, p)
    nvcc = shutil.which("nvcc")
    if nvcc:
        toolkit_base = str(Path(nvcc).parent.parent)
        for sub in ("lib64", "lib", "compat"):
            p = os.path.join(toolkit_base, sub)
            if p not in candidates:
                candidates.insert(0, p)
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
        if any(f.startswith("libnvshmem") for f in files):
            if p not in dirs:
                dirs.append(p)

    return dirs


def _find_cudnn_root() -> str | None:
    """Return the cuDNN root directory if nvidia-cudnn-cuXX is pip-installed.

    Prefers the CUDA-versioned package (``nvidia-cudnn-cu12`` etc.) over
    the generic ``nvidia-cudnn``, and returns the highest CUDA-major match
    when multiple ``-cuXX`` variants are present.
    """
    roots: list[tuple[int, str]] = []
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"] or ""
            if name.startswith("nvidia-cudnn-cu"):
                try:
                    loc = dist.locate_file("")
                except Exception:
                    loc = dist._path.parent
                root = str(loc / "nvidia/cudnn")
                if os.path.isdir(os.path.join(root, "lib")):
                    # Parse CUDA major from suffix e.g. nvidia-cudnn-cu12 → 12
                    cuda_major = 0
                    suffix = name.split("nvidia-cudnn-cu")[-1]
                    try:
                        cuda_major = int(suffix)
                    except ValueError:
                        pass
                    roots.append((cuda_major, root))
            elif name == "nvidia-cudnn" and not roots:
                # Only fall back to generic nvidia-cudnn if no -cuXX found
                try:
                    loc = dist.locate_file("")
                except Exception:
                    loc = dist._path.parent
                root = str(loc / "nvidia/cudnn")
                if os.path.isdir(os.path.join(root, "lib")):
                    roots.append((0, root))
    except Exception:
        pass
    if not roots:
        return None
    # Return the package with the highest CUDA major (prefers -cuXX over generic)
    roots.sort(key=lambda x: x[0], reverse=True)
    return roots[0][1]


def _ensure_pip_package(pip_name: str) -> bool:
    """Install *pip_name* via ``pip install --user`` if not already present.

    Returns ``True`` if the package was already installed or installed
    successfully.  Returns ``False`` on any failure (no network, no
    permissions, etc.) — *never raises*.
    """
    import sys

    if not pip_name or not pip_name.strip():
        return False

    # Check if already installed via importlib.metadata
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"] or ""
            if name.replace("-", "_") == pip_name.replace("-", "_"):
                return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", pip_name],
            capture_output=True, timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


# Libraries that NVIDIA HPC benchmark binaries may require at runtime.
KNOWN_MISSING_LIBS = ("libcublas", "libmpi", "libnccl", "libnvshmem")


def _detect_gpu_model(gpu_index: int) -> str:
    """Return GPU model name for the given device index.

    Uses ``nvidia-smi`` first, falls back to CuPy, then to ``'unknown'``.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(gpu_index)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode()
        return str(name)
    except Exception:
        pass
    return "unknown"


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
    elif lib_name == "libnvshmem":
        nvshmem_found = _find_nvshmem_libs()
        cuda_ver = _guess_cuda_major()
        detail += (
            "\n\nNVSHMEM library not found. The HPL binary requires "
            "libnvshmem_host.so at runtime.\n"
            "  Install via pip (recommended):\n"
            f"    pip install --user nvidia-nvshmem-cu{cuda_ver}\n"
            "  Or install a system NVSHMEM package:\n"
            "    Ubuntu/Debian: sudo apt install nvidia-nvshmem\n"
            f"  Searched pip nvidia-nvshmem-cu* paths: {nvshmem_found}\n"
            "  Check: pip list | grep nvshmem"
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
    cuda_libs.extend(_find_cudnn_libs())
    cuda_libs.extend(_find_nvshmem_libs())

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
            # OpenMPI needs OPAL_PREFIX to find its plugins at runtime
            if "OPAL_PREFIX" not in env:
                env["OPAL_PREFIX"] = mpi_base
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

    # Set CUDNN_ROOT if nvidia-cudnn-cuXX is pip-installed
    cudnn_root = _find_cudnn_root()
    if cudnn_root:
        env["CUDNN_ROOT"] = cudnn_root

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
