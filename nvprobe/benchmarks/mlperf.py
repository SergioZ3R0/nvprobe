"""MLPerf Inference benchmark wrapper using cmx4mlperf (cr run-mlperf)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult, subprocess_env


def _find_mlperf_cmd() -> str | None:
    """Find cr or cmx command (from cmx4mlperf)."""
    for name in ["cr", "cmx"]:
        path = shutil.which(name)
        if path:
            return path
    # Also check ~/.local/bin where --user installs go
    import os
    local_bin = os.path.expanduser("~/.local/bin")
    for name in ["cr", "cmx"]:
        path = os.path.join(local_bin, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


class MlperfBenchmark(BaseBenchmark):
    """Wrapper around MLPerf inference via cmx4mlperf."""

    name = "mlperf"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        model = self.params.get("model", "resnet50")
        framework = self.params.get("framework", "onnxruntime")
        scenario = self.params.get("scenario", "Offline")
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

        cmd_name = "cr" if os.path.basename(mlperf_cmd) == "cr" else "cmx"

        if cmd_name == "cr":
            cmd = [
                mlperf_cmd, "run-mlperf,inference,_find-performance,_full",
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
                mlperf_cmd, "run", "script",
                "run-mlperf,inference,_find-performance,_full",
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
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False, error=f"{exc}\n{stderr}".strip(),
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return shell commands for this benchmark (without SBATCH headers)."""
        model = self.params.get("model", "resnet50")
        framework = self.params.get("framework", "onnxruntime")
        scenario = self.params.get("scenario", "Offline")
        category = self.params.get("category", "edge")
        implementation = self.params.get("implementation", "reference")
        test_query_count = self.params.get("test_query_count", 100)

        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

cr run-mlperf,inference,_find-performance,_full \\
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
