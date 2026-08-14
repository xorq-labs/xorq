"""Bad-input paths through the build-archive validators.

These used to raise a bare ``AssertionError`` -- no type a caller could catch,
and nothing at all under ``python -O``. Each test pins the exception a user
hands bad input actually gets.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from xorq.catalog.zip_utils import (
    BuildZip,
    extract_build_zip_to,
    write_zip,
)
from xorq.common.exceptions import XorqInputError
from xorq.ibis_yaml.enums import REQUIRED_ARCHIVE_NAMES


TEST_WHEEL_NAME = "xorq_test-0.0.0-py3-none-any.whl"


def valid_archive(path: Path) -> Path:
    return write_zip(
        path, dict.fromkeys((*REQUIRED_ARCHIVE_NAMES, TEST_WHEEL_NAME), b"")
    )


def test_build_zip_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.zip"
    with pytest.raises(FileNotFoundError, match="nope.zip"):
        BuildZip(missing)


def test_build_zip_bad_suffix_raises_input_error(tmp_path: Path) -> None:
    path = valid_archive(tmp_path / "build.zip").rename(tmp_path / "build.tar")
    with pytest.raises(XorqInputError, match="suffix"):
        BuildZip(path)


def test_extract_build_zip_rejects_multiple_top_level_dirs(tmp_path: Path) -> None:
    path = tmp_path / "two-roots.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("first/expr.yaml", b"")
        zf.writestr("second/expr.yaml", b"")
    target = tmp_path / "extracted"
    target.mkdir()
    with pytest.raises(XorqInputError, match="exactly one top-level directory"):
        extract_build_zip_to(path, target)
