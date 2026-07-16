"""MLPerf Inference benchmark wrapper using cmx4mlperf (cr run-mlperf)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


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
    deps = ["loguru"]
    env = subprocess_env()
    for pkg in deps:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", pkg],
                capture_output=True, timeout=60, env=env,
            )


class MlperfBenchmark(BaseBenchmark):
    """Wrapper around MLPerf inference via cmx4mlperf."""

    name = "mlperf"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        model = self.params.get("model", "resnet50")
        framework = self.params.get("framework", "onnxruntime")
        scenario = _normalize_scenario(self.params.get("scenario", "Offline"))
        category = self.params.get("category", "edge")
        implementation = self.params.get("implementation", "reference")
        test_query_count = self.params.get("test_query_count", 100)

        mlperf_cmd = _find_mlperf_cmd()
        if not mlperf_cmd:
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False,
                error="MLPerf CLI not found. Install with: pip install --user cmx4mlperf",
            )

        _ensure_mlperf_deps()

        cmd_name = os.path.basename(mlperf_cmd)

        if cmd_name in ("cr", "mlcr"):
            cmd = [
                mlperf_cmd, "run-mlperf,inference",
                f"--model={model}",
                f"--implementation={implementation}",
                f"--framework={framework}",
                f"--category={category}",
                f"--scenario={scenario}",
                "--execution_mode=test",
                "--device=cuda",
                f"--test_query_count={test_query_count}",
                "--quiet",
            ]
        else:
            cmd = [
                mlperf_cmd, "run",
                "run-mlperf,inference",
                f"--model={model}",
                f"--implementation={implementation}",
                f"--framework={framework}",
                f"--category={category}",
                f"--scenario={scenario}",
                "--execution_mode=test",
                "--device=cuda",
                f"--test_query_count={test_query_count}",
                "--quiet",
            ]

        env = subprocess_env()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

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
            error_msg = f"{exc}\n{stderr}".strip()
            if "Permission denied" in error_msg:
                error_msg += (
                    "\n\nFix: pip install --user loguru"
                    "\nThen re-run: nvprobe run --config <your-config> --local"
                )
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False, error=error_msg,
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return shell commands for this benchmark (without SBATCH headers)."""
        model = self.params.get("model", "resnet50")
        framework = self.params.get("framework", "onnxruntime")
        scenario = _normalize_scenario(self.params.get("scenario", "Offline"))
        category = self.params.get("category", "edge")
        implementation = self.params.get("implementation", "reference")
        test_query_count = self.params.get("test_query_count", 100)

        mlperf_cmd = _find_mlperf_cmd()
        cmd_name = os.path.basename(mlperf_cmd) if mlperf_cmd else "cr"
        if cmd_name in ("cr", "mlcr"):
            mlperf_line = f"{cmd_name} run-mlperf,inference \\"
        else:
            mlperf_line = f"{cmd_name} run \\"

        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

pip install --user loguru 2>/dev/null || true

{mlperf_line}
    --model={model} \\
    --implementation={implementation} \\
    --framework={framework} \\
    --category={category} \\
    --scenario={scenario} \\
    --execution_mode=test \\
    --device=cuda \\
    --test_query_count={test_query_count} \\
    --quiet
"""
