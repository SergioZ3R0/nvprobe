"""Slurm integration — per-node job submission, monitoring, and result collection.

Each compute node runs nvprobe locally, creating its own database.
After all jobs complete, per-node databases are merged into a unified one,
making the Slurm workflow identical to the local workflow for reporting.
"""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from nvprobe.config import RunConfig


@dataclass
class SlurmJob:
    """Represents a submitted Slurm node job."""

    job_id: str
    node_index: int
    script_path: str
    output_path: str | None = None
    status: str = "pending"


class SlurmManager:
    """Manages Slurm job lifecycle: generate, submit, monitor, collect, merge."""

    def __init__(self, config: RunConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.scripts_dir = output_dir / "slurm_scripts"
        self.jobs_dir = output_dir / "slurm_jobs"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_file = self.jobs_dir / "jobs.json"
        self._jobs = self._load_jobs()

    # --- Public API ---

    def generate_scripts(self) -> list[Path]:
        """Generate one sbatch script per node.

        Each script runs ``nvprobe run --local`` on its allocated node,
        writing results to a per-node subdirectory.
        """
        slurm = self.config.slurm
        num_nodes = max(1, slurm.nodes)
        gpus_per_node = max(1, slurm.gpus_per_node)

        # Copy config to output dir so compute nodes can access it
        config_src = getattr(self.config, "_source_path", None)
        if config_src is None:
            raise RuntimeError(
                "Slurm config must be loaded from a YAML file via load_config()"
            )
        config_src = Path(config_src)
        config_dst = self.output_dir / "cluster_config.yaml"
        shutil.copy2(config_src, config_dst)

        scripts: list[Path] = []
        for node_idx in range(num_nodes):
            header = self._build_header(node_idx, gpus_per_node)
            body = self._build_body(node_idx, config_dst)
            full_script = header + "\n" + body

            script_path = self.scripts_dir / f"nvprobe_node_{node_idx}.sh"
            script_path.write_text(full_script, encoding="utf-8")
            scripts.append(script_path)

        print(f"Generated {len(scripts)} node script(s) in {self.scripts_dir}")
        return scripts

    def submit_all(self, scripts: list[Path] | None = None) -> list[SlurmJob]:
        """Submit all generated scripts (or provided list) to Slurm."""
        if scripts is None:
            scripts = sorted(self.scripts_dir.glob("nvprobe_node_*.sh"))

        for script in scripts:
            job = self._submit_script(script)
            if job:
                self._jobs.append(job)

        self._save_jobs()
        print(f"Submitted {len(self._jobs)} job(s)")
        return self._jobs

    def monitor(self, poll_interval: int = 30) -> None:
        """Poll Slurm until all jobs complete."""
        if not self._jobs:
            print("No jobs to monitor.")
            return

        print(f"Monitoring {len(self._jobs)} job(s) (poll every {poll_interval}s)...")
        while True:
            running = self._get_running_jobs()
            completed = len(self._jobs) - len(running)
            print(f"  {completed}/{len(self._jobs)} completed", end="\r")

            if not running:
                print()
                break

            time.sleep(poll_interval)

        self._update_job_statuses()
        self._save_jobs()

    def collect_results(self) -> dict[str, Any]:
        """Collect results from completed jobs and merge databases."""
        merged_db_path = self.output_dir / "benchmarks.db"
        env_info = merge_databases(
            self.output_dir, merged_db_path, self.config.name,
            self.config.description,
        )
        return {
            "merged_db": str(merged_db_path),
            "gpus": len(env_info.get("gpus", [])),
            "nodes": len(self._jobs),
        }

    # --- Job persistence ---

    def _load_jobs(self) -> list[SlurmJob]:
        """Load jobs from disk if the file exists."""
        if self._jobs_file and self._jobs_file.exists():
            try:
                data = json.loads(self._jobs_file.read_text(encoding="utf-8"))
                return [SlurmJob(**j) for j in data]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"WARNING: could not load jobs file: {exc}")
        return []

    def _save_jobs(self) -> None:
        """Save all jobs to disk."""
        data = [asdict(j) for j in self._jobs]
        self._jobs_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # --- Internal helpers ---

    def _build_header(self, node_idx: int, gpus_per_node: int) -> str:
        """Build Slurm SBATCH header for a single-node job."""
        slurm = self.config.slurm
        job_name = f"nvprobe-n{node_idx}"
        output_file = str(self.jobs_dir / f"{job_name}_%j.out").replace("\\", "/")
        error_file = str(self.jobs_dir / f"{job_name}_%j.err").replace("\\", "/")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={output_file}",
            f"#SBATCH --error={error_file}",
            f"#SBATCH --partition={slurm.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --gpus={gpus_per_node}",
            f"#SBATCH --time={slurm.time_limit}",
        ]

        if slurm.account:
            lines.append(f"#SBATCH --account={slurm.account}")
        if slurm.exclude:
            lines.append(f"#SBATCH --exclude={slurm.exclude}")
        for arg in slurm.extra_args:
            lines.append(f"#SBATCH {arg}")

        lines.extend([
            "",
            "module purge 2>/dev/null || true",
            "module load cuda 2>/dev/null || true",
            "",
            f"echo \"=== nvProbe Node {node_idx} ===\"",
            "echo \"Job ID: $SLURM_JOB_ID\"",
            "echo \"Node: $(hostname)\"",
            "echo \"GPUs: $SLURM_GPUS_ON_NODE\"",
            "echo \"==========================\"",
            "",
        ])

        return "\n".join(lines)

    def _build_body(self, node_idx: int, config_path: Path) -> str:
        """Build the body: run nvprobe locally on this node."""
        node_output = self.output_dir / f"node_{node_idx}"
        return f"""# Navigate to output directory and run
cd {self.output_dir.resolve()}

python3 -m nvprobe.cli run \\
    --config {config_path.resolve()} \\
    --local \\
    --output {node_output.resolve()}

echo "nvprobe completed on $(hostname)"
"""

    def _submit_script(self, script_path: Path) -> SlurmJob | None:
        """Submit a single sbatch script and return SlurmJob."""
        try:
            proc = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, check=True,
            )
            job_id = proc.stdout.strip().split()[-1]

            # Parse node index from filename: nvprobe_node_0.sh → 0
            stem = script_path.stem
            node_idx = int(stem.rsplit("_", 1)[-1])

            job_name = f"nvprobe-n{node_idx}"
            output_path_str = str(self.jobs_dir / f"{job_name}_{job_id}.out")
            return SlurmJob(
                job_id=job_id,
                node_index=node_idx,
                script_path=str(script_path),
                output_path=output_path_str,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"WARNING: failed to submit {script_path.name}: {exc}")
            return None

    def _get_running_jobs(self) -> list[str]:
        """Return list of still-running job IDs."""
        try:
            proc = subprocess.run(
                ["squeue", "--noheader", "--format=%i", "--state=R,PD,CG"],
                capture_output=True, text=True, check=True,
            )
            running_ids = {line.strip() for line in proc.stdout.strip().splitlines() if line.strip()}
            return [j.job_id for j in self._jobs if j.job_id in running_ids]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def _update_job_statuses(self) -> None:
        """Update job statuses from sacct."""
        for job in self._jobs:
            try:
                proc = subprocess.run(
                    ["sacct", "--noheader", "-j", job.job_id, "-o", "State", "--parsable2"],
                    capture_output=True, text=True, check=True,
                )
                states = [s.strip() for s in proc.stdout.strip().splitlines() if s.strip()]
                if states:
                    job.status = states[0]
            except (subprocess.CalledProcessError, FileNotFoundError):
                job.status = "unknown"


def merge_databases(
    output_dir: Path,
    merged_db_path: Path,
    run_name: str = "slurm-run",
    run_description: str = "",
) -> dict[str, Any]:
    """Merge per-node databases into a single unified database.

    Returns the merged environment fingerprint.
    """
    # Find all node databases
    node_dirs = sorted(output_dir.glob("node_*"))
    node_db_paths = [d / "benchmarks.db" for d in node_dirs if (d / "benchmarks.db").exists()]
    node_env_paths = [d / "environment.json" for d in node_dirs if (d / "environment.json").exists()]

    if not node_db_paths:
        print("WARNING: no node databases found to merge")
        return {"gpus": []}

    # Build merged environment from all node fingerprints
    merged_env = _merge_environments(node_env_paths)

    # Create merged database
    merged_db = sqlite3.connect(str(merged_db_path))
    _init_merged_db(merged_db)
    run_id = _create_merged_run(merged_db, run_name, run_description, merged_env)

    # Copy results from each node with re-indexed GPU indices
    gpu_offset = 0
    for db_path in node_db_paths:
        node_dir = db_path.parent
        env_path = node_dir / "environment.json"
        gpu_count = _count_gpus_in_env(env_path)
        node_db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        rows = node_db.execute(
            "SELECT * FROM results ORDER BY benchmark, gpu_index, precision"
        ).fetchall()

        for row in rows:
            rd = dict(row)
            new_gpu = rd["gpu_index"] + gpu_offset
            identity = _make_identity(
                rd["benchmark"], new_gpu, rd["precision"], rd["batch_size"],
            )
            merged_db.execute(
                """INSERT OR IGNORE INTO results
                   (run_id, identity, benchmark, gpu_model, gpu_index,
                    precision, batch_size, metrics, raw_output,
                    success, error, elapsed_sec, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, identity, rd["benchmark"], rd["gpu_model"],
                    new_gpu, rd["precision"], rd["batch_size"],
                    rd["metrics"], rd["raw_output"], rd["success"],
                    rd["error"], rd["elapsed_sec"], rd["created_at"],
                ),
            )

        node_db.close()
        gpu_offset += gpu_count

    merged_db.commit()
    merged_db.close()

    print(f"Merged {len(node_db_paths)} node database(s) → {merged_db_path}")
    print(f"  Total GPUs: {len(merged_env.get('gpus', []))}")
    return merged_env


def _merge_environments(env_paths: list[Path]) -> dict[str, Any]:
    """Merge environment JSON files from multiple nodes.

    Concatenates GPU lists with hostname-prefixed model names and re-indexed indices.
    """
    merged: dict[str, Any] = {
        "timestamp": "",
        "hostname": "merged",
        "kernel": "",
        "driver_version": "",
        "cuda_version": "",
        "gpus": [],
    }

    global_idx = 0
    for path in env_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        host = data.get("hostname", "unknown")
        if not merged["driver_version"]:
            merged["driver_version"] = data.get("driver_version", "")
        if not merged["cuda_version"]:
            merged["cuda_version"] = data.get("cuda_version", "")
        if not merged["kernel"]:
            merged["kernel"] = data.get("kernel", "")
        if not merged["timestamp"]:
            merged["timestamp"] = data.get("timestamp", "")

        for gpu in data.get("gpus", []):
            gpu_copy = copy.deepcopy(gpu)
            gpu_copy["model"] = f"{gpu_copy.get('model', 'unknown')} ({host})"
            gpu_copy["index"] = global_idx
            merged["gpus"].append(gpu_copy)
            global_idx += 1

    return merged


def _init_merged_db(db: sqlite3.Connection) -> None:
    """Create tables in merged database if they don't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            environment TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            identity TEXT UNIQUE,
            benchmark TEXT NOT NULL,
            gpu_model TEXT NOT NULL,
            gpu_index INTEGER NOT NULL,
            precision TEXT DEFAULT '',
            batch_size INTEGER DEFAULT 1,
            metrics TEXT DEFAULT '{}',
            raw_output TEXT DEFAULT '',
            success INTEGER DEFAULT 1,
            error TEXT DEFAULT '',
            elapsed_sec REAL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)
    db.commit()


def _create_merged_run(
    db: sqlite3.Connection, name: str, description: str, env: dict[str, Any],
) -> int:
    """Insert a run entry into the merged database."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cur = db.execute(
        "INSERT INTO runs (name, description, environment, created_at) VALUES (?, ?, ?, ?)",
        (name, description, json.dumps(env, default=str), now),
    )
    db.commit()
    return cur.lastrowid or 0


def _count_gpus_in_env(env_path: Path) -> int:
    """Return the number of GPUs reported in an environment JSON file."""
    try:
        data = json.loads(env_path.read_text(encoding="utf-8"))
        return len(data.get("gpus", []))
    except (json.JSONDecodeError, OSError):
        return 0


def _make_identity(benchmark: str, gpu_index: int, precision: str, batch_size: int) -> str:
    """Build a unique identity string for deduplication."""
    return f"{benchmark}:{gpu_index}:{precision}:{batch_size}"
