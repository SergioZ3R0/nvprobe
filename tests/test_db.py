"""Tests for CUDA version parsing from nvidia-smi banner (two banner formats)."""

import re

from nvprobe.db import _parse_int_or_none, _query_nvlink_status

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


class TestParseIntOrNone:
    def test_plain_number(self) -> None:
        assert _parse_int_or_none("42") == 42

    def test_na(self) -> None:
        assert _parse_int_or_none("N/A") is None

    def test_bracketed_na(self) -> None:
        assert _parse_int_or_none("[N/A]") is None

    def test_bracketed_enabled(self) -> None:
        assert _parse_int_or_none("[Enabled]") is None

    def test_bracketed_number(self) -> None:
        assert _parse_int_or_none("[18]") == 18

    def test_empty(self) -> None:
        assert _parse_int_or_none("") is None

    def test_unknown(self) -> None:
        assert _parse_int_or_none("Unknown") is None

    def test_not_supported(self) -> None:
        assert _parse_int_or_none("Not Supported") is None

    def test_garbage(self) -> None:
        assert _parse_int_or_none("foo") is None


class TestQueryNvlinkStatus:
    def test_active_links(self) -> None:
        output = (
            "GPU 0: NVIDIA B200 (UUID: GPU-xxx)\n"
            "         Link 0: 26.562 GB/s\n"
            "         Link 1: 26.562 GB/s\n"
            "         Link 2: <inactive>\n"
        )
        result = _query_nvlink_status_for_test(output)
        assert result == 2

    def test_all_inactive(self) -> None:
        output = (
            "GPU 0: NVIDIA B200 (UUID: GPU-xxx)\n"
            "         Link 0: <inactive>\n"
            "         Link 1: <inactive>\n"
        )
        result = _query_nvlink_status_for_test(output)
        assert result is None

    def test_no_links(self) -> None:
        output = "GPU 0: NVIDIA B200 (UUID: GPU-xxx)\n"
        result = _query_nvlink_status_for_test(output)
        assert result is None


def _query_nvlink_status_for_test(output: str) -> int | None:
    active = 0
    for line in output.splitlines():
        if re.match(r"\s*Link \d+:", line) and "<inactive>" not in line:
            active += 1
    return active if active > 0 else None
