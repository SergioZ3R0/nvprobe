"""Tests for the missing-shared-library diagnostic machinery."""

from nvprobe.benchmarks.base import _diagnose_missing_lib, _ensure_pip_package


def test_diagnose_libnccl_does_not_raise() -> None:
    """_diagnose_missing_lib("libnccl", ...) must return a string without exceptions."""
    detail = (
        "error while loading shared libraries: libnccl.so.2: "
        "cannot open shared object file: No such file or directory"
    )
    result = _diagnose_missing_lib("libnccl", detail)
    assert isinstance(result, str)
    assert len(result) > len(detail)
    assert "pip install" in result.lower()
    assert "nvidia-nccl-cu" in result


def test_diagnose_libcublas_does_not_raise() -> None:
    detail = "libcublas.so: cannot open shared object file"
    result = _diagnose_missing_lib("libcublas", detail)
    assert isinstance(result, str)
    assert len(result) > len(detail)


def test_diagnose_libmpi_does_not_raise() -> None:
    detail = "libmpi.so.12: cannot open shared object file"
    result = _diagnose_missing_lib("libmpi", detail)
    assert isinstance(result, str)
    assert len(result) > len(detail)


def test_diagnose_libnvshmem_does_not_raise() -> None:
    detail = "libnvshmem_host.so.3: cannot open shared object file"
    result = _diagnose_missing_lib("libnvshmem", detail)
    assert isinstance(result, str)
    assert len(result) > len(detail)
    assert "pip install" in result.lower()
    assert "nvidia-nvshmem-cu" in result


def test_diagnose_unknown_lib_passthrough() -> None:
    """Unknown lib names should be returned unchanged (no KeyError / crash)."""
    detail = "some random error"
    result = _diagnose_missing_lib("libnonexistent", detail)
    assert result == detail


def test_ensure_pip_package_already_installed() -> None:
    """A known-installed package (e.g. 'pytest') should return True."""
    result = _ensure_pip_package("pytest")
    assert result is True


def test_ensure_pip_package_not_installed_returns_false() -> None:
    """A nonexistent package should return False without raising."""
    result = _ensure_pip_package("nonexistent-package-12345-xyz")
    assert result is False


def test_ensure_pip_package_never_raises() -> None:
    """Even bizarre inputs should never raise."""
    result = _ensure_pip_package("")
    assert result is False
    result = _ensure_pip_package("\x00invalid")
    assert result is False
