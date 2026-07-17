"""MLPerf Inference benchmark wrapper using cmx4mlperf (cr run-mlperf)."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from nvprobe.benchmarks.base import (
    BaseBenchmark, BenchmarkResult, _ensure_pip_package,
    _find_cudnn_root, subprocess_env,
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


def _register_cudnn_with_mlcr(mlperf_cmd: str, cudnn_root: str) -> str | None:
    """Pre-register a pip-installed cuDNN with mlcr's cache so it is found
    during the MLPerf pipeline dependency resolution.

    Returns ``None`` on success or an error message on failure.
    """
    cudnn_lib = os.path.join(cudnn_root, "lib")
    if not os.path.isdir(cudnn_lib):
        return f"cuDNN lib directory not found: {cudnn_lib}"

    env = os.environ.copy()
    env["CUDNN_ROOT"] = cudnn_root
    # Pre-set CM_TMP_PATH so step 2 of get-cudnn/customize.py finds
    # libcudnn.so immediately (single-directory shortcut).
    env["CM_TMP_PATH"] = cudnn_lib

    cmd = [mlperf_cmd, "get,cudnn,nvidia", f"--input={cudnn_root}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        if proc.returncode == 0:
            return None
        stderr = (proc.stderr or "")[-300:]
        return f"mlcr get,cudnn,nvidia returned {proc.returncode}: {stderr}"
    except FileNotFoundError:
        return f"mlcr command not found: {mlperf_cmd}"
    except subprocess.TimeoutExpired:
        return "mlcr get,cudnn timed out"
    except Exception as exc:
        return f"mlcr get,cudnn failed: {exc}"


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

        env = subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # Point mlcr to pip-installed cuDNN if available
        cudnn_root = _find_cudnn_root()
        if cudnn_root:
            env["CUDNN_ROOT"] = cudnn_root
            env["CM_TMP_PATH"] = os.path.join(cudnn_root, "lib")
            # Pre-register with mlcr's cache so dependency resolution finds it
            reg_err = _register_cudnn_with_mlcr(mlperf_cmd, cudnn_root)
            if reg_err:
                env["CM_CUDA_PATH_LIB_CUDNN"] = os.path.join(cudnn_root, "lib")
                env["CM_CUDA_PATH_LIB_CUDNN_EXISTS"] = "yes"

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200, check=True,
                env=env,
            )
            return BenchmarkResult(
                benchmark=self.name,
                gpu_model="unknown",
                gpu_index=gpu_index,
                precision=precision,
                batch_size=batch_size,
                metrics={
                    "model": model,
                    "framework": framework,
                    "scenario": scenario,
                },
                raw_output=proc.stdout,
            )
        except FileNotFoundError:
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error="MLPerf CLI not found. Install with: pip install --user cmx4mlperf",
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            # Extract only the meaningful error lines, skip verbose mlcr logs
            error_lines = []
            for line in stderr.splitlines():
                line_stripped = line.strip()
                if any(kw in line_stripped.lower() for kw in (
                    "error", "failed", "permission denied", "cudnn",
                    "not found", "exception", "traceback",
                )):
                    error_lines.append(line_stripped)
            detail = "\n".join(error_lines[-10:]) if error_lines else stderr.strip()[-300:]

            if "cudnn" in detail.lower() or "cudnn" in stderr.lower():
                cuda_ver = "13"
                try:
                    from nvprobe.benchmarks.base import _guess_cuda_major
                    cuda_ver = _guess_cuda_major()
                except Exception:
                    pass
                pip_cmd = f"pip install --user nvidia-cudnn-cu{cuda_ver}"
                detail = (
                    "cuDNN not detected by MLPerf pipeline.\n"
                    f"  1. (recommended) Install the pip package: {pip_cmd}\n"
                    "     Then run 'nvprobe run' again (auto-detected).\n"
                    "  2. Download cuDNN tar from https://developer.nvidia.com/cudnn\n"
                    "     then register it:\n"
                    f"       mlcr get,cudnn,nvidia --tar_file=/path/to/cudnn-linux-*.tar.xz\n"
                    "  3. If cuDNN is already installed system-wide, set:\n"
                    "       export CUDNN_ROOT=/path/to/cudnn"
                )
            elif "Permission denied" in detail:
                detail += "\n\nFix: pip install --user loguru"

            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
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
