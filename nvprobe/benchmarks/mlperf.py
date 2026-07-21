"""MLPerf Inference benchmark wrapper using cmx4mlperf (cr run-mlperf)."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from nvprobe.benchmarks.base import (
    BaseBenchmark, BenchmarkResult, _detect_gpu_model, _ensure_pip_package,
    _find_cudnn_root, subprocess_env,
)

_CUDNN_REGISTERED_SENTINEL = os.path.join(
    str(Path.home()), ".nvprobe", ".cudnn_registered"
)


def _find_mlperf_cmd() -> str | None:
    """Find mlcr, cr, or cmx command (from cmx4mlperf)."""
    for name in ["mlcr", "cr", "cmx"]:
        path = shutil.which(name)
        if path:
            return path
    local_bin = os.path.expanduser("~/.local/bin")
    for name in ["mlcr", "cr", "cmx"]:
        path = os.path.join(local_bin, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


_VALID_SCENARIOS = {"offline", "server", "singlestream", "multistream"}
_SCENARIO_ALIASES = {"inference": "offline", "perf": "offline", "default": "offline"}


def _normalize_scenario(scenario: str) -> str:
    """Map common user mistakes to valid MLPerf scenario names."""
    lower = scenario.lower().strip()
    if lower in _SCENARIO_ALIASES:
        return _SCENARIO_ALIASES[lower].capitalize()
    if lower in _VALID_SCENARIOS:
        return scenario.capitalize()
    return scenario  # pass through, mlcr will report the error


def _ensure_mlperf_deps() -> None:
    """Pre-install dependencies cr needs but can't install itself (no root)."""
    _ensure_pip_package("loguru")


def _register_cudnn_once(mlperf_cmd: str, cudnn_root: str) -> None:
    """Register pip-installed cuDNN with mlcr's cache (idempotent).

    Runs ``mlcr get,cudnn,nvidia --input=<cudnn_root>`` once and creates
    a sentinel file so it is not repeated in subsequent calls within the
    same benchmark run.
    """
    if os.path.isfile(_CUDNN_REGISTERED_SENTINEL):
        return
    cudnn_lib = os.path.join(cudnn_root, "lib")
    if not os.path.isdir(cudnn_lib):
        return
    try:
        subprocess.run(
            [mlperf_cmd, "get,cudnn,nvidia", f"--input={cudnn_root}"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        pass
    # Create sentinel even on failure — avoid hammering mlcr on every retry
    try:
        sentinel_dir = os.path.dirname(_CUDNN_REGISTERED_SENTINEL)
        os.makedirs(sentinel_dir, exist_ok=True)
        Path(_CUDNN_REGISTERED_SENTINEL).touch()
    except Exception:
        pass


def _clear_mlc_cuda_cache(mlperf_cmd: str) -> None:
    """Clear MLC's cached CUDA detection so mlcr re-detects the real CUDA version."""
    try:
        subprocess.run(
            [mlperf_cmd, "rm", "cache", "--tags=get,cuda", "-f"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        pass


def _detect_cuda_major() -> int | None:
    """Detect CUDA major version from nvcc or nvidia-smi."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, check=True,
            )
            for line in out.stdout.splitlines():
                if "release" in line:
                    ver = line.split("release")[-1].strip().rstrip(",").split(",")[0]
                    return int(ver.split(".")[0])
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        parts = out.stdout.strip().split(".")
        if parts:
            major = int(parts[0])
            if major >= 535:
                return 12
            elif major >= 525:
                return 11
            return 11
    except Exception:
        pass
    return None


def _check_cudnn_cuda_compat() -> str:
    """Check if installed cuDNN CUDA variant matches system CUDA version.

    Returns a warning string if mismatch detected, empty string otherwise.
    """
    cuda_major = _detect_cuda_major()
    if cuda_major is None:
        return ""
    cudnn_cuda: int | None = None
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata["Name"] or ""
            if name.startswith("nvidia-cudnn-cu"):
                suffix = name.split("nvidia-cudnn-cu")[-1]
                try:
                    cudnn_cuda = int(suffix)
                except ValueError:
                    pass
                break
    except Exception:
        pass
    if cudnn_cuda is not None and cudnn_cuda != cuda_major:
        return (
            f"WARNING: nvidia-cudnn-cu{cudnn_cuda} is installed but system CUDA "
            f"version is {cuda_major}. This may cause onnxruntime to fall back "
            f"to CPU. Install: pip install --user nvidia-cudnn-cu{cuda_major}"
        )
    return ""


_CPU_FALLBACK_PATTERNS = (
    "failed to create cudaexecutionprovider",
    "libcudnn.so.",  # catches libcudnn.so.9, libcudnn.so.8, etc.
    "cuda execution provider is not available",
    "onnxruntime.*cuda.*not.*available",
    "fallback.*cpu",
    "could not load library.*cudnn",
    "cuda error",
)


def _detect_cpu_fallback(text: str) -> bool:
    """Return True if *text* indicates onnxruntime fell back to CPU."""
    lower = text.lower()
    return any(p in lower for p in _CPU_FALLBACK_PATTERNS)


class MlperfBenchmark(BaseBenchmark):
    """Wrapper around MLPerf inference via cmx4mlperf."""

    name = "mlperf"

    def _build_cmd(self, mlperf_cmd: str, scenario: str, **kwargs) -> list[str]:
        cmd_name = os.path.basename(mlperf_cmd)
        mode = kwargs.get("mode", "test")
        if mode == "find_performance":
            mode_suffix = ",_find-performance"
        elif mode == "full":
            mode_suffix = ",_find-performance,_full"
        else:
            mode_suffix = ""

        if cmd_name in ("cr", "mlcr"):
            base = [mlperf_cmd, f"run-mlperf,inference{mode_suffix}"]
        else:
            base = [mlperf_cmd, "run", f"run-mlperf,inference{mode_suffix}"]

        return base

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        model = self.params.get("model", "resnet50")
        framework = self.params.get("framework", "onnxruntime")
        scenario = _normalize_scenario(self.params.get("scenario", "Offline"))
        category = self.params.get("category", "edge")
        implementation = self.params.get("implementation", "reference")
        test_query_count = self.params.get("test_query_count", 100)
        mode = self.params.get("mode", "test")
        custom_batch_size = self.params.get("batch_size")

        mlperf_cmd = _find_mlperf_cmd()
        if not mlperf_cmd:
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error="MLPerf CLI not found. Install with: pip install --user cmx4mlperf",
            )

        _ensure_mlperf_deps()

        # Fix 1: clear MLC's stale CUDA cache before invoking mlcr
        _clear_mlc_cuda_cache(mlperf_cmd)

        # Fix 2: check cuDNN vs system CUDA version compatibility
        compat_warning = _check_cudnn_cuda_compat()

        gpu_model = _detect_gpu_model(gpu_index)

        cmd = self._build_cmd(mlperf_cmd, scenario, mode=mode)
        cmd.extend([
            f"--model={model}",
            f"--implementation={implementation}",
            f"--framework={framework}",
            f"--category={category}",
            f"--scenario={scenario}",
            f"--execution_mode={mode}",
            "--device=cuda",
            f"--test_query_count={test_query_count}",
            "--quiet",
        ])
        if custom_batch_size is not None:
            cmd.append(f"--batch_size={custom_batch_size}")

        # Register pip-installed cuDNN with mlcr's cache (once per run).
        # Also set LD_LIBRARY_PATH so sub-processes find libcudnn at runtime.
        cudnn_root = _find_cudnn_root()
        cudnn_lib: str | None = None
        if cudnn_root:
            _register_cudnn_once(mlperf_cmd, cudnn_root)
            lib = os.path.join(cudnn_root, "lib")
            if os.path.isdir(lib):
                cudnn_lib = lib

        try:
            env = subprocess_env()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            if cudnn_lib:
                env["CUDNN_ROOT"] = cudnn_root
                env["MLC_CUDA_PATH_LIB_CUDNN"] = cudnn_lib
                env["MLC_CUDA_PATH_INCLUDE_CUDNN"] = os.path.join(cudnn_root, "include")
                env["LD_LIBRARY_PATH"] = (
                    f"{cudnn_lib}:{env['LD_LIBRARY_PATH']}"
                    if env.get("LD_LIBRARY_PATH") else cudnn_lib
                )
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200, check=True,
                env=env,
            )

            # Fix 3: detect CPU fallback even when exit code is 0
            full_output = proc.stdout + "\n" + proc.stderr
            cpu_warning = ""
            if _detect_cpu_fallback(full_output):
                cpu_warning = (
                    "WARNING: MLPerf benchmark likely ran on CPU, not GPU. "
                    "onnxruntime's CUDAExecutionProvider failed to load "
                    "(check cuDNN installation / CUDA version compatibility). "
                    "The reported performance numbers are for CPU, not GPU."
                )

            metrics: dict[str, Any] = {
                "model": model,
                "framework": framework,
                "scenario": scenario,
            }
            if compat_warning:
                metrics["_compat_warning"] = compat_warning
            if cpu_warning:
                metrics["_cpu_warning"] = cpu_warning

            return BenchmarkResult(
                benchmark=self.name,
                gpu_model=gpu_model,
                gpu_index=gpu_index,
                precision=precision,
                batch_size=batch_size,
                metrics=metrics,
                raw_output=proc.stdout,
            )
        except FileNotFoundError:
            return BenchmarkResult(
                benchmark=self.name, gpu_model=gpu_model, gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error="MLPerf CLI not found. Install with: pip install --user cmx4mlperf",
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            stdout = getattr(exc, "stdout", "") or ""

            # --- debugging: print the exact command and full stderr ---
            print(f"    [DEBUG] Failed command: {' '.join(cmd)}")
            print(f"    [DEBUG] Full stderr ({len(stderr)} chars):")
            for line in stderr.splitlines():
                print(f"      | {line}")

            # --- real cuDNN error patterns (not just any mention of "cudnn") ---
            _CUDNN_ERROR_PATTERNS = (
                "libcudnn.so", "cudnn is not", "cudnn not found",
                "cudnn.*not found", "cudnn_root is not set",
                "cannot open shared object", "no module named.*cudnn",
                "cudnn.*failed", "cudnn.*error",
            )
            stderr_lower = stderr.lower()
            has_real_cudnn_error = any(
                p in stderr_lower or p in stderr
                for p in _CUDNN_ERROR_PATTERNS
            )

            # Extract meaningful error lines, skipping verbose mlcr logs
            error_lines = []
            for line in stderr.splitlines():
                line_stripped = line.strip()
                if any(kw in line_stripped.lower() for kw in (
                    "error", "failed", "permission denied",
                    "not found", "exception", "traceback",
                )):
                    error_lines.append(line_stripped)
            detail = "\n".join(error_lines[-10:]) if error_lines else stderr.strip()[-300:]

            if has_real_cudnn_error:
                detail = (
                    "cuDNN not detected by MLPerf pipeline.\n"
                    "  mlcr's sub-scripts do not inherit parent environment\n"
                    "  variables (CUDNN_ROOT, LD_LIBRARY_PATH).\n"
                    "  Options:\n"
                    f"  1. Register pip-installed cuDNN with mlcr:\n"
                    f"       mlcr get,cudnn,nvidia --input=$(python3 -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')\n"
                    "     Then run 'nvprobe run' again.\n"
                    "  2. Download cuDNN tar from https://developer.nvidia.com/cudnn\n"
                    "     and register it:\n"
                    f"       mlcr get,cudnn,nvidia --tar_file=/path/to/cudnn-linux-*.tar.xz\n"
                    "  3. Install cuDNN system-wide (RPM/deb) into the CUDA\n"
                    "     toolkit directory."
                )
            elif "Permission denied" in detail:
                detail += "\n\nFix: pip install --user loguru"

            # Append compat warning if present
            full_output = stdout + "\n" + stderr
            if _detect_cpu_fallback(full_output) and compat_warning:
                detail = compat_warning + "\n\n" + detail
            elif _detect_cpu_fallback(full_output):
                detail = (
                    "WARNING: MLPerf benchmark likely ran on CPU, not GPU.\n"
                    "  onnxruntime's CUDAExecutionProvider failed to load.\n\n"
                ) + detail
            elif compat_warning:
                detail = compat_warning + "\n\n" + detail

            return BenchmarkResult(
                benchmark=self.name, gpu_model=gpu_model, gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False, error=detail or f"{exc}",
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return shell commands for this benchmark (without SBATCH headers)."""
        model = self.params.get("model", "resnet50")
        framework = self.params.get("framework", "onnxruntime")
        scenario = _normalize_scenario(self.params.get("scenario", "Offline"))
        category = self.params.get("category", "edge")
        implementation = self.params.get("implementation", "reference")
        test_query_count = self.params.get("test_query_count", 100)
        mode = self.params.get("mode", "test")
        custom_batch_size = self.params.get("batch_size")

        mlperf_cmd = _find_mlperf_cmd()
        cmd_name = os.path.basename(mlperf_cmd) if mlperf_cmd else "cr"
        mode_suffix = ""
        if mode == "find_performance":
            mode_suffix = ",_find-performance"
        elif mode == "full":
            mode_suffix = ",_find-performance,_full"
        if cmd_name in ("cr", "mlcr"):
            mlperf_line = f"{cmd_name} run-mlperf,inference{mode_suffix} \\"
        else:
            mlperf_line = f"{cmd_name} run run-mlperf,inference{mode_suffix} \\"

        batch_line = ""
        if custom_batch_size is not None:
            batch_line = f"    --batch_size={custom_batch_size} \\\n"

        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

pip install --user loguru 2>/dev/null || true

{mlperf_line}
    --model={model} \\
    --implementation={implementation} \\
    --framework={framework} \\
    --category={category} \\
    --scenario={scenario} \\
    --execution_mode={mode} \\
    --device=cuda \\
    --test_query_count={test_query_count} \\
{batch_line}    --quiet
"""
