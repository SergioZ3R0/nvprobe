"""Tests that MPI commands include --mca pmix isolated for OpenMPI/Slurm compatibility."""

from unittest.mock import MagicMock, mock_open, patch

from nvprobe.benchmarks.hpl import _run_hpl_size
from nvprobe.benchmarks.hpcg import _run_hpcg_size


def test_hpl_uses_mca_pmix_isolated_when_mpi_run() -> None:
    """_run_hpl_size must add --mca pmix isolated when mpi_run is given."""
    with (
        patch("nvprobe.benchmarks.hpl.subprocess.run") as mock_run,
        patch("nvprobe.benchmarks.hpl.tempfile.TemporaryDirectory") as mock_tmp,
        patch("nvprobe.benchmarks.hpl.open", mock_open()),
        patch("nvprobe.benchmarks.hpl._generate_hpl_dat", return_value=""),
    ):
        mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
        mock_run.return_value = MagicMock(stdout="WC00C00R2     1024   1   1   5.12345", returncode=0)

        _run_hpl_size(
            binary="/fake/xhpl",
            n=1024,
            mpi_run="/usr/bin/mpirun",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpu_index=0,
            precision="fp64",
            batch_size=0,
        )

    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "/usr/bin/mpirun"
    assert cmd[1:4] == ["--mca", "pmix", "isolated"]
    assert "-np" in cmd
    assert "1" in cmd
    assert "/fake/xhpl" in cmd


def test_hpl_skips_mca_when_no_mpi_run() -> None:
    """_run_hpl_size without mpi_run must not include --mca."""
    with (
        patch("nvprobe.benchmarks.hpl.subprocess.run") as mock_run,
        patch("nvprobe.benchmarks.hpl.tempfile.TemporaryDirectory") as mock_tmp,
        patch("nvprobe.benchmarks.hpl.open", mock_open()),
        patch("nvprobe.benchmarks.hpl._generate_hpl_dat", return_value=""),
    ):
        mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
        mock_run.return_value = MagicMock(stdout="WC00C00R2     1024   1   1   5.12345", returncode=0)

        _run_hpl_size(
            binary="/fake/xhpl",
            n=1024,
            mpi_run=None,
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpu_index=0,
            precision="fp64",
            batch_size=0,
        )

    args, kw = mock_run.call_args
    cmd = args[0]
    assert cmd == ["/fake/xhpl"]
    assert kw["env"].get("OMPI_MCA_pmix") == "isolated"


def test_hpcg_uses_mca_pmix_isolated_when_mpi_run() -> None:
    """_run_hpcg_size must add --mca pmix isolated when mpi_run is given."""
    with patch("nvprobe.benchmarks.hpcg.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Result: 5.12345 GFLOP/s", returncode=0)

        _run_hpcg_size(
            binary="/fake/xhpcg",
            size=128,
            mpi_run="/usr/bin/mpirun",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpu_index=0,
            precision="fp64",
            batch_size=0,
        )

    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "/usr/bin/mpirun"
    assert cmd[1:4] == ["--mca", "pmix", "isolated"]
    assert "-np" in cmd
    assert "1" in cmd
    assert "/fake/xhpcg" in cmd


def test_hpcg_skips_mca_when_no_mpi_run() -> None:
    """_run_hpcg_size without mpi_run must not include --mca."""
    with patch("nvprobe.benchmarks.hpcg.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Result: 5.12345 GFLOP/s", returncode=0)

        _run_hpcg_size(
            binary="/fake/xhpcg",
            size=128,
            mpi_run=None,
            env={"CUDA_VISIBLE_DEVICES": "0"},
            gpu_index=0,
            precision="fp64",
            batch_size=0,
        )

    args, kw = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "/fake/xhpcg"
    assert kw["env"].get("OMPI_MCA_pmix") == "isolated"
