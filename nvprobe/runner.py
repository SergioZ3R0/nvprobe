"""Benchmark runner — orchestrates execution via local or Slurm."""

from __future__ import annotations

import copy
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

    all_ok = True

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

            bench_ok = _run_single_benchmark(
                db, run_id, bench_cfg, bench_cls, gpus, config, dry_run,
            )
            if not bench_ok:
                all_ok = False
    finally:
        db.close()
        if all_ok:
            print(f"\nAll benchmarks completed. Results saved to {output_dir}")
        else:
            print(f"\nResults saved to {output_dir}")


def _get_sizes(bench_cfg: Any, size_keys: list[str]) -> list:
    """Get the list of sizes to iterate over from the benchmark config."""
    for key in size_keys:
        vals = bench_cfg.params.get(key)
        if vals:
            return list(vals)
    return []


def _run_and_log(
    db: Database, run_id: int, bench_cls: type, bench_cfg: Any,
    gpu_index: int, precision: str, batch_size: int, dry_run: bool,
) -> bool:
    """Run a single benchmark combination and log to DB.

    Returns True if the benchmark succeeded (even if individual results fail),
    False only on unexpected exceptions.
    """
    params = copy.deepcopy(bench_cfg.params)
    instance = bench_cls(params)
    label = f"{bench_cfg.name} gpu={gpu_index} prec={precision} bs={batch_size}"

    if dry_run:
        print(f"  [dry-run] {label}")
        return True

    print(f"  {label} ... ", end="", flush=True)
    t0 = time.monotonic()
    try:
        result = instance.run_local(gpu_index, precision, batch_size)
        elapsed = time.monotonic() - t0
        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"{status} ({elapsed:.1f}s)")
        db.insert_result(run_id, result, elapsed)
        return True
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"ERROR: {exc} ({elapsed:.1f}s)")
        from nvprobe.benchmarks.base import BenchmarkResult
        result = BenchmarkResult(
            benchmark=bench_cfg.name, gpu_model="unknown",
            gpu_index=gpu_index, precision=precision, batch_size=batch_size,
            success=False, error=f"Unhandled exception: {exc}",
        )
        db.insert_result(run_id, result, elapsed)
        return False


def _run_single_benchmark(
    db: Database, run_id: int, bench_cfg: Any, bench_cls: type,
    gpus: list[dict], config: RunConfig, dry_run: bool,
) -> bool:
    """Run a single benchmark across all parameter combinations.

    Iterates over GPUs, precisions, batch sizes, and benchmark-specific
    size keys (e.g. problem_sizes for HPL). Each combination is wrapped
    in try/except so a single failure doesn't abort the rest.

    Returns True if all runs completed (even with failures), False on internal error.
    """
    print(f"Running: {bench_cfg.name}")
    benchmark = bench_cls(bench_cfg.params)
    all_ok = True

    if not benchmark.uses_precision_batch:
        size_keys: list[str] = getattr(benchmark, "size_keys", [])
        sizes = _get_sizes(bench_cfg, size_keys)
        if not sizes:
            sizes = [None]

        for gpu in gpus:
            gpu_index = gpu["index"]
            for size in sizes:
                params = copy.deepcopy(bench_cfg.params)
                if size is not None and size_keys:
                    params[size_keys[0]] = [size]
                instance = bench_cls(params)
                label = f"{bench_cfg.name} gpu={gpu_index} size={size}"

                if dry_run:
                    print(f"  [dry-run] {label}")
                    continue

                print(f"  {label} ... ", end="", flush=True)
                t0 = time.monotonic()
                try:
                    result = instance.run_local(gpu_index, "fp32", 1)
                    elapsed = time.monotonic() - t0
                    status = "OK" if result.success else f"FAIL: {result.error}"
                    print(f"{status} ({elapsed:.1f}s)")
                    db.insert_result(run_id, result, elapsed)
                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    print(f"ERROR: {exc} ({elapsed:.1f}s)")
                    from nvprobe.benchmarks.base import BenchmarkResult
                    result = BenchmarkResult(
                        benchmark=bench_cfg.name, gpu_model="unknown",
                        gpu_index=gpu_index, precision="fp32", batch_size=1,
                        success=False, error=f"Unhandled exception: {exc}",
                    )
                    db.insert_result(run_id, result, elapsed)
                    all_ok = False
    else:
        for precision in config.precisions:
            for batch_size in config.batch_sizes:
                for gpu in gpus:
                    gpu_index = gpu["index"]
                    ok = _run_and_log(
                        db, run_id, bench_cls, bench_cfg,
                        gpu_index, precision, batch_size, dry_run,
                    )
                    if not ok:
                        all_ok = False

    return all_ok


def _run_cmd(cmd: list[str]) -> str:
    """Run a command and return stdout, raising on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def _save_json(path: Path, data: Any) -> None:
    """Write data as pretty JSON."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
