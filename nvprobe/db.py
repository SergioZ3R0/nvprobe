"""SQLite database for benchmark results with CSV/JSON export."""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nvprobe.benchmarks.base import BenchmarkResult


class Database:
    """SQLite storage for benchmark runs and results."""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row

    def init(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                environment TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                benchmark TEXT NOT NULL,
                gpu_model TEXT,
                gpu_index INTEGER,
                precision TEXT,
                batch_size INTEGER,
                metrics TEXT,
                raw_output TEXT,
                success INTEGER,
                error TEXT,
                elapsed_seconds REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
            CREATE INDEX IF NOT EXISTS idx_results_benchmark ON results(benchmark);
        """)
        self._conn.commit()

    def create_run(self, name: str, description: str, environment: dict[str, Any]) -> int:
        """Create a new run and return its ID."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO runs (name, description, environment, created_at) VALUES (?, ?, ?, ?)",
            (name, description, json.dumps(environment, default=str), now),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def insert_result(self, run_id: int, result: BenchmarkResult, elapsed: float) -> int:
        """Insert a benchmark result and return its ID."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO results
               (run_id, benchmark, gpu_model, gpu_index, precision, batch_size,
                metrics, raw_output, success, error, elapsed_seconds, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                result.benchmark,
                result.gpu_model,
                result.gpu_index,
                result.precision,
                result.batch_size,
                json.dumps(result.metrics, default=str),
                result.raw_output,
                1 if result.success else 0,
                result.error,
                elapsed,
                now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_runs(self) -> list[dict[str, Any]]:
        """Return all runs."""
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_results(self, run_id: int) -> list[dict[str, Any]]:
        """Return all results for a given run."""
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY benchmark, gpu_index, precision",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_results_by_benchmark(self, run_id: int, benchmark: str) -> list[dict[str, Any]]:
        """Return results filtered by benchmark name."""
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? AND benchmark = ? ORDER BY gpu_index, precision, batch_size",
            (run_id, benchmark),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_results_by_gpu(self, run_id: int, gpu_model: str) -> list[dict[str, Any]]:
        """Return results filtered by GPU model."""
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? AND gpu_model = ? ORDER BY benchmark, precision, batch_size",
            (run_id, gpu_model),
        ).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, run_id: int, output_path: Path) -> Path:
        """Export results to CSV file."""
        results = self.get_results(run_id)
        if not results:
            raise ValueError(f"No results found for run {run_id}")

        csv_path = output_path.with_suffix(".csv")
        fieldnames = ["benchmark", "gpu_model", "gpu_index", "precision", "batch_size",
                       "elapsed_seconds", "success", "error", "metrics"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {k: r[k] for k in fieldnames if k in r}
                writer.writerow(row)

        return csv_path

    def export_json(self, run_id: int, output_path: Path) -> Path:
        """Export results to JSON file."""
        run = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        results = self.get_results(run_id)
        data = {
            "run": dict(run),
            "results": results,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return json_path

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def _parse_mib(value: str) -> int:
    """Parse a memory value that might include 'MiB' suffix."""
    try:
        return int(value)
    except ValueError:
        return int(value.replace("MiB", "").strip())


def _nvidia_smi_query(*fields: str, gpu_index: int | None = None) -> list[str]:
    """Run nvidia-smi with given query fields, return list of per-row strings."""
    cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}",
           "--format=csv,noheader,nounits"]
    if gpu_index is not None:
        cmd.extend(["-i", str(gpu_index)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def fingerprint_environment() -> dict[str, Any]:
    """Capture detailed environment fingerprint for reproducibility."""
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": _run_cmd_safe(["hostname"]).strip(),
        "kernel": _run_cmd_safe(["uname", "-r"]).strip(),
        "driver_version": "",
        "cuda_version": "",
        "nvidia_smi_full": "",
        "gpus": [],
        "python_version": _run_cmd_safe([sys.executable, "--version"]).strip(),
    }

    smi = _run_cmd_safe(["nvidia-smi"])
    info["nvidia_smi_full"] = smi

    # Try combined query first (fast path)
    all_rows = _nvidia_smi_query(
        "driver_version", "cuda_version", "name", "index", "memory.total", "pci.bus_id",
    )
    if all_rows:
        for row in all_rows:
            parts = row.split(",")
            if len(parts) >= 6:
                info["driver_version"] = parts[0].strip()
                info["cuda_version"] = parts[1].strip()
                info["gpus"].append({
                    "model": parts[2].strip(),
                    "index": int(parts[3].strip()),
                    "memory_total_mb": _parse_mib(parts[4].strip()),
                    "pci_bus_id": parts[5].strip(),
                })

    # Fallback: per-GPU queries (handles commas in GPU names gracefully)
    if not info["gpus"]:
        for i in range(64):  # upper bound
            rows = _nvidia_smi_query("name", "index", "memory.total", "pci.bus_id", gpu_index=i)
            if not rows:
                break
            parts = rows[0].split(",")
            if len(parts) >= 4:
                info["gpus"].append({
                    "model": parts[0].strip(),
                    "index": int(parts[1].strip()),
                    "memory_total_mb": _parse_mib(parts[2].strip()),
                    "pci_bus_id": parts[3].strip(),
                })

    # CuPy fallback
    if not info["gpus"]:
        try:
            import cupy as cp
            count = cp.cuda.runtime.getDeviceCount()
            for i in range(count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props.get("name", b"unknown")
                if isinstance(name, bytes):
                    name = name.decode()
                try:
                    with cp.cuda.Device(i):
                        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                    mem_mb = total_bytes // (1024 * 1024)
                except Exception:
                    mem_mb = 0
                info["gpus"].append({
                    "model": str(name),
                    "index": i,
                    "memory_total_mb": mem_mb,
                    "pci_bus_id": "",
                })
                if not info["driver_version"]:
                    info["driver_version"] = f"via cupy (CUDA {cp.cuda.runtime.runtimeGetVersion()})"
        except Exception:
            pass

    return info


def _run_cmd_safe(cmd: list[str]) -> str:
    """Run a command safely, returning empty string on failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
