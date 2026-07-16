"""Slurm integration — job submission, monitoring, and result collection."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nvprobe.benchmarks import BENCHMARK_REGISTRY
from nvprobe.config import RunConfig


@dataclass
class SlurmJob:
    """Represents a submitted Slurm job."""

    job_id: str
    benchmark: str
    gpu_index: int
    precision: str
    batch_size: int
    script_path: Path
    output_path: Path | None = None
    status: str = "pending"


class SlurmManager:
    """Manages Slurm job lifecycle: generate, submit, monitor, collect."""

    def __init__(self, config: RunConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.scripts_dir = output_dir / "slurm_scripts"
        self.jobs_dir = output_dir / "slurm_jobs"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: list[SlurmJob] = []

    def generate_scripts(self) -> list[Path]:
        """Generate sbatch scripts for all enabled benchmarks x configs."""
        env_info = _detect_environment()
        gpus = env_info.get("gpus", [])
        if not gpus:
            print("WARNING: no GPUs detected, generating scripts for GPU 0 only")
            gpus = [{"index": 0}]

        scripts: list[Path] = []
        for bench_cfg in self.config.benchmarks:
            if not bench_cfg.enabled:
                continue
            bench_cls = BENCHMARK_REGISTRY.get(bench_cfg.name)
            if bench_cls is None:
                continue

            benchmark = bench_cls(bench_cfg.params)

            for precision in self.config.precisions:
                for batch_size in self.config.batch_sizes:
                    for gpu in gpus:
                        gpu_index = gpu["index"]
                        script_content = benchmark.build_slurm_script(gpu_index, precision, batch_size)
                        header = self._build_header(bench_cfg.name, gpu_index, precision, batch_size)
                        full_script = header + "\n" + script_content

                        script_name = f"{bench_cfg.name}_gpu{gpu_index}_{precision}_bs{batch_size}.sh"
                        script_path = self.scripts_dir / script_name
                        script_path.write_text(full_script, encoding="utf-8")
                        scripts.append(script_path)

        print(f"Generated {len(scripts)} Slurm scripts in {self.scripts_dir}")
        return scripts

    def submit_all(self, scripts: list[Path] | None = None) -> list[SlurmJob]:
        """Submit all generated scripts (or provided list) to Slurm."""
        if scripts is None:
            scripts = list(self.scripts_dir.glob("*.sh"))

        for script in scripts:
            job = self._submit_script(script)
            if job:
                self._jobs.append(job)

        print(f"Submitted {len(self._jobs)} jobs")
        return self._jobs

    def monitor(self, poll_interval: int = 30) -> None:
        """Poll Slurm until all jobs complete."""
        if not self._jobs:
            return

        print(f"Monitoring {len(self._jobs)} jobs (poll every {poll_interval}s)...")
        while True:
            running = self._get_running_jobs()
            completed = len(self._jobs) - len(running)
            print(f"  {completed}/{len(self._jobs)} completed", end="\r")

            if not running:
                print()
                break

            time.sleep(poll_interval)

        self._update_job_statuses()

    def collect_results(self) -> dict[str, Any]:
        """Collect output from completed jobs."""
        results: dict[str, Any] = {}

        for job in self._jobs:
            if job.output_path and job.output_path.exists():
                output = job.output_path.read_text(encoding="utf-8")
                key = f"{job.benchmark}_gpu{job.gpu_index}_{job.precision}_bs{job.batch_size}"
                results[key] = {
                    "job_id": job.job_id,
                    "benchmark": job.benchmark,
                    "gpu_index": job.gpu_index,
                    "precision": job.precision,
                    "batch_size": job.batch_size,
                    "output": output,
                    "status": job.status,
                }

        return results

    def _build_header(self, benchmark: str, gpu_index: int, precision: str, batch_size: int) -> str:
        """Build Slurm SBATCH header from config."""
        slurm = self.config.slurm
        job_name = f"nvprobe-{benchmark}-gpu{gpu_index}-{precision}-bs{batch_size}"
        output_file = str(self.jobs_dir / f"{job_name}_%j.out").replace("\\", "/")
        error_file = str(self.jobs_dir / f"{job_name}_%j.err").replace("\\", "/")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={output_file}",
            f"#SBATCH --error={error_file}",
            f"#SBATCH --partition={slurm.partition}",
            f"#SBATCH --nodes={slurm.nodes}",
            f"#SBATCH --ntasks=1",
            f"#SBATCH --gpus={slurm.gpus_per_node}",
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
            "echo \"=== nvProbe Slurm Job ===\"",
            "echo \"Job ID: $SLURM_JOB_ID\"",
            "echo \"Node: $(hostname)\"",
            "echo \"GPUs: $SLURM_GPUS_ON_NODE\"",
            "echo \"==========================\"",
            "",
        ])

        return "\n".join(lines)

    def _submit_script(self, script_path: Path) -> SlurmJob | None:
        """Submit a single sbatch script and return SlurmJob."""
        try:
            proc = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, check=True,
            )
            # sbatch output: "Submitted batch job 12345"
            job_id = proc.stdout.strip().split()[-1]
            parts = script_path.stem.split("_")
            # Build the job_name exactly as _build_header does
            job_name = f"nvprobe-{parts[0]}-gpu{parts[1].replace('gpu', '')}-{parts[2]}-bs{parts[3].replace('bs', '')}"
            return SlurmJob(
                job_id=job_id,
                benchmark=parts[0],
                gpu_index=int(parts[1].replace("gpu", "")),
                precision=parts[2],
                batch_size=int(parts[3].replace("bs", "")),
                script_path=script_path,
                output_path=self.jobs_dir / f"{job_name}_{job_id}.out",
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


def _detect_environment() -> dict[str, Any]:
    """Quick GPU detection for Slurm script generation."""
    info: dict[str, Any] = {"gpus": []}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info["gpus"].append({"index": int(parts[0]), "model": parts[1]})
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info
