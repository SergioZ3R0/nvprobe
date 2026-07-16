"""YAML configuration loader and schema for nvProbe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _expand_user(val: Any) -> Any:
    """Expand ~ in string values."""
    if isinstance(val, str):
        return str(Path(val).expanduser())
    return val


@dataclass
class BenchmarkConfig:
    """A single benchmark to run."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUConfig:
    """GPU hardware filter for a run."""

    models: list[str] = field(default_factory=list)  # e.g. ["L40S", "B200"]
    min_count: int = 1


@dataclass
class SlurmConfig:
    """Slurm submission settings."""

    enabled: bool = True
    partition: str = "gpu"
    account: str = ""
    time_limit: str = "01:00:00"
    gpus_per_node: int = 1
    nodes: int = 1
    exclude: str = ""
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """Top-level config for an nvProbe run."""

    name: str = "benchmark-run"
    description: str = ""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    benchmarks: list[BenchmarkConfig] = field(default_factory=list)
    precisions: list[str] = field(default_factory=lambda: ["fp32", "fp16", "int8"])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 32, 64, 128])
    environment: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> RunConfig:
    """Load a YAML config file and return a validated RunConfig."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> RunConfig:
    """Parse raw YAML dict into RunConfig dataclass."""
    gpu_raw = raw.get("gpu", {})
    gpu = GPUConfig(
        models=gpu_raw.get("models", []),
        min_count=gpu_raw.get("min_count", 1),
    )

    slurm_raw = raw.get("slurm", {})
    slurm = SlurmConfig(
        enabled=slurm_raw.get("enabled", True),
        partition=slurm_raw.get("partition", "gpu"),
        account=slurm_raw.get("account", ""),
        time_limit=slurm_raw.get("time_limit", "01:00:00"),
        gpus_per_node=slurm_raw.get("gpus_per_node", 1),
        nodes=slurm_raw.get("nodes", 1),
        exclude=slurm_raw.get("exclude", ""),
        extra_args=slurm_raw.get("extra_args", []),
    )

    benchmarks = []
    for b in raw.get("benchmarks", []):
        params = {k: _expand_user(v) for k, v in b.get("params", {}).items()}
        benchmarks.append(BenchmarkConfig(
            name=b["name"],
            enabled=b.get("enabled", True),
            params=params,
        ))

    return RunConfig(
        name=raw.get("name", "benchmark-run"),
        description=raw.get("description", ""),
        gpu=gpu,
        slurm=slurm,
        benchmarks=benchmarks,
        precisions=raw.get("precisions", ["fp32", "fp16", "int8"]),
        batch_sizes=raw.get("batch_sizes", [1, 32, 64, 128]),
        environment=raw.get("environment", {}),
    )
