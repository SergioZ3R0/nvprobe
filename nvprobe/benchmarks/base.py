"""Base class for all benchmark modules."""

from __future__ import annotations

import os
import site
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def subprocess_env() -> dict[str, str]:
    """Return an environment dict that includes user site-packages."""
    env = os.environ.copy()
    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(user_site):
        pythonpath = env.get("PYTHONPATH", "")
        parts = [p for p in pythonpath.split(os.pathsep) if p]
        if user_site not in parts:
            parts.insert(0, user_site)
        env["PYTHONPATH"] = os.pathsep.join(parts)
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
