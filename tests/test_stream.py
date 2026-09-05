"""Tests for STREAM benchmark — command construction and output parsing."""

import json
import sys
from unittest.mock import MagicMock, patch

from nvprobe.benchmarks.stream import StreamBenchmark


def test_stream_run_local_constructs_correct_command() -> None:
    """StreamBenchmark.run_local must invoke the CUDA subprocess with correct args."""
    fake_output = json.dumps({
        "benchmark": "stream",
        "gpu_model": "NVIDIA B200",
        "gpu_index": 0,
        "precision": "fp32",
        "iterations": 20,
        "sizes": [10000000],
        "metrics": {
            "10000000": {
                "copy": {"mean": 800.0, "min": 790.0, "max": 810.0, "std": 5.0, "avg_ms": 0.1},
                "scale": {"mean": 750.0, "min": 740.0, "max": 760.0, "std": 4.0, "avg_ms": 0.1},
                "add": {"mean": 700.0, "min": 690.0, "max": 710.0, "std": 3.0, "avg_ms": 0.1},
                "triad": {"mean": 650.0, "min": 640.0, "max": 660.0, "std": 3.0, "avg_ms": 0.1},
                "n_elements": 10000000,
                "total_bytes_mb": 38.15,
            }
        },
    })

    with patch("nvprobe.benchmarks.stream.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=fake_output, returncode=0)
        bench = StreamBenchmark({"sizes": [10000000], "iterations": 20})
        result = bench.run_local(gpu_index=0, precision="fp32", batch_size=1)

    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "nvprobe.benchmarks._cuda.stream_test" in cmd
    assert "--gpu" in cmd
    assert "0" in cmd
    assert "--sizes" in cmd
    assert "10000000" in cmd
    assert "--iterations" in cmd
    assert "20" in cmd
    assert "--precision" in cmd
    assert "fp32" in cmd
    assert result.success is True
    assert result.metrics["10000000"]["copy"]["mean"] == 800.0


def test_stream_run_local_handles_error() -> None:
    """StreamBenchmark.run_local must return failure on subprocess error."""
    import subprocess as sp

    with patch("nvprobe.benchmarks.stream.subprocess.run") as mock_run:
        mock_run.side_effect = sp.CalledProcessError(1, "cmd", stderr="CUDA not available")
        bench = StreamBenchmark({"sizes": [10000000], "iterations": 20})
        result = bench.run_local(gpu_index=0, precision="fp32", batch_size=1)

    assert result.success is False
    assert "CUDA not available" in result.error


def test_stream_build_slurm_script_contains_required_elements() -> None:
    """build_slurm_script must export CUDA_VISIBLE_DEVICES and call the right module."""
    bench = StreamBenchmark({"sizes": [10000000, 100000000], "iterations": 10})
    script = bench.build_slurm_script(gpu_index=2, precision="fp64", batch_size=1)

    assert "export CUDA_VISIBLE_DEVICES=2" in script
    assert "nvprobe.benchmarks._cuda.stream_test" in script
    assert "--sizes 10000000,100000000" in script
    assert "--iterations 10" in script
    assert "--precision fp64" in script


def test_stream_uses_default_params() -> None:
    """StreamBenchmark must fall back to sensible defaults when params are empty."""
    with patch("nvprobe.benchmarks.stream.subprocess.run") as mock_run:
        fake_output = json.dumps({
            "benchmark": "stream",
            "gpu_model": "RTX 4090",
            "gpu_index": 0,
            "precision": "fp32",
            "iterations": 20,
            "sizes": [10000000, 100000000],
            "metrics": {},
        })
        mock_run.return_value = MagicMock(stdout=fake_output, returncode=0)
        bench = StreamBenchmark({})
        bench.run_local(gpu_index=0, precision="fp32", batch_size=1)

    args, _ = mock_run.call_args
    cmd = args[0]
    # Default sizes: [10000000, 100000000]
    assert "10000000,100000000" in cmd
    # Default iterations: 20
    assert "--iterations" in cmd
    assert "20" in cmd
