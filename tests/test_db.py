"""Tests for CUDA version parsing from nvidia-smi banner (two banner formats)."""

import re

SMI_OLD = (
    "NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0\n"
    "GPU 00000000:00:00.0  Off  ..."
)

SMI_NEW = (
    "NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3\n"
    "GPU 00000000:00:00.0  Off  ..."
)


def _parse_cuda_from_banner(smi: str) -> str | None:
    m = re.search(r'CUDA (?:UMD )?Version:\s*([\d.]+)', smi)
    return m.group(1) if m else None


def _parse_driver_from_banner(smi: str) -> str | None:
    m = re.search(r'(?:Driver|KMD) Version:\s*([\d.]+)', smi)
    return m.group(1) if m else None


def test_cuda_version_old_format() -> None:
    assert _parse_cuda_from_banner(SMI_OLD) == "13.0"


def test_cuda_version_new_format() -> None:
    assert _parse_cuda_from_banner(SMI_NEW) == "13.3"


def test_cuda_version_no_match() -> None:
    assert _parse_cuda_from_banner("no version here") is None


def test_driver_version_old_format() -> None:
    assert _parse_driver_from_banner(SMI_OLD) == "580.126.09"


def test_driver_version_new_format() -> None:
    assert _parse_driver_from_banner(SMI_NEW) == "610.43.02"
