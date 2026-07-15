"""MLPerf Inference/Training benchmark wrapper."""

from __future__ import annotations

import subprocess
from typing import Any

from nvprobe.benchmarks.base import BaseBenchmark, BenchmarkResult


class MlperfBenchmark(BaseBenchmark):
    """Wrapper around the MLPerf benchmark suite."""

    name = "mlperf"

    def run_local(self, gpu_index: int, precision: str, batch_size: int) -> BenchmarkResult:
        scenario = self.params.get("scenario", "inference")
        mode = self.params.get("mode", "performance")
        dataset = self.params.get("dataset", "openimages")

        cmd = [
            "python3", "-m", "mlperf_" + scenario,
            "--gpu", str(gpu_index),
            "--mode", mode,
            "--dataset", dataset,
            "--batch-size", str(batch_size),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200, check=True,
            )
            return BenchmarkResult(
                benchmark=self.name,
                gpu_model="unknown",
                gpu_index=gpu_index,
                precision=precision,
                batch_size=batch_size,
                metrics={"scenario": scenario, "mode": mode},
                raw_output=proc.stdout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return BenchmarkResult(
                benchmark=self.name, gpu_model="unknown", gpu_index=gpu_index,
                precision=precision, batch_size=batch_size,
                success=False, error=str(exc),
            )

    def build_slurm_script(self, gpu_index: int, precision: str, batch_size: int) -> str:
        """Return shell commands for this benchmark (without SBATCH headers)."""
        scenario = self.params.get("scenario", "inference")
        mode = self.params.get("mode", "performance")
        dataset = self.params.get("dataset", "openimages")
        return f"""export CUDA_VISIBLE_DEVICES={gpu_index}

python3 -m mlperf_{scenario} \\
    --gpu 0 \\
    --mode {mode} \\
    --dataset {dataset} \\
    --batch-size {batch_size}
"""
