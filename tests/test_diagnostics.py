"""Tests for the missing-shared-library diagnostic machinery."""

import re

from nvprobe.benchmarks.base import _diagnose_missing_lib


def test_diagnose_libnccl_does_not_raise() -> None:
    """_diagnose_missing_lib(\"libnccl\", ...) must return a string without exceptions."""
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


def test_diagnose_unknown_lib_passthrough() -> None:
    """Unknown lib names should be returned unchanged (no KeyError / crash)."""
    detail = "some random error"
    result = _diagnose_missing_lib("libnonexistent", detail)
    assert result == detail
