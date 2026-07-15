"""SQLite database for benchmark results."""

from __future__ import annotations

import json
import sqlite3
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

    def close(self) -> None:
        self._conn.close()
