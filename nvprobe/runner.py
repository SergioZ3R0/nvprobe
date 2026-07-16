"""Benchmark runner — orchestrates execution via local or Slurm."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nvprobe.benchmarks import BENCHMARK_REGISTRY
from nvprobe.config import RunConfig, load_config
from nvprobe.db import Database, fingerprint_environment


def detect_environment() -> dict[str, Any]:
    """Detect GPU environment: driver version, CUDA version, GPU models, etc."""
    return fingerprint_environment()


def run_benchmarks(config_path: Path, output_dir: Path, local: bool = False, dry_run: bool = False) -> None:
    """Run all enabled benchmarks from config, saving results to output_dir."""
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_info = detect_environment()
    _save_json(output_dir / "environment.json", env_info)

    db = Database(output_dir / "benchmarks.db")
    db.init()

    run_id = db.create_run(config.name, config.description, env_info)

    try:
        gpus = env_info.get("gpus", [])
        if not gpus:
            print("WARNING: No GPUs detected via nvidia-smi. Assuming GPU 0.")
            gpus = [{"index": 0, "model": "unknown"}]
        else:
            print(f"Detected {len(gpus)} GPU(s): {', '.join(g['model'] for g in gpus)}")

        for bench_cfg in config.benchmarks:
            if not bench_cfg.enabled:
                continue

            bench_cls = BENCHMARK_REGISTRY.get(bench_cfg.name)
            if bench_cls is None:
                print(f"WARNING: unknown benchmark '{bench_cfg.name}', skipping")
                continue

            benchmark = bench_cls(bench_cfg.params)
            print(f"Running: {bench_cfg.name}")

            if not benchmark.uses_precision_batch:
                for gpu in gpus:
                    gpu_index = gpu["index"]
                    if dry_run:
                        print(f"  [dry-run] {bench_cfg.name} gpu={gpu_index}")
                        continue
                    print(f"  {bench_cfg.name} gpu={gpu_index} ... ", end="", flush=True)
                    t0 = time.monotonic()
                    result = benchmark.run_local(gpu_index, "fp32", 1)
                    elapsed = time.monotonic() - t0
                    status = "OK" if result.success else f"FAIL: {result.error}"
                    print(f"{status} ({elapsed:.1f}s)")
                    db.insert_result(run_id, result, elapsed)
            else:
                for precision in config.precisions:
                    for batch_size in config.batch_sizes:
                        for gpu in gpus:
                            gpu_index = gpu["index"]

                            if dry_run:
                                print(f"  [dry-run] {bench_cfg.name} gpu={gpu_index} prec={precision} bs={batch_size}")
                                continue

                            print(f"  {bench_cfg.name} gpu={gpu_index} prec={precision} bs={batch_size} ... ", end="", flush=True)
                            t0 = time.monotonic()
                            result = benchmark.run_local(gpu_index, precision, batch_size)
                            elapsed = time.monotonic() - t0

                            status = "OK" if result.success else f"FAIL: {result.error}"
                            print(f"{status} ({elapsed:.1f}s)")

                            db.insert_result(run_id, result, elapsed)
    finally:
        db.close()
        print(f"\nResults saved to {output_dir}")


def _run_cmd(cmd: list[str]) -> str:
    """Run a command and return stdout, raising on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def _save_json(path: Path, data: Any) -> None:
    """Write data as pretty JSON."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
