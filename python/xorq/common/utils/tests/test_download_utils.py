"""Bad-input paths through the archive download/extract helpers.

``extract_zip`` refuses to overwrite and requires a single-rooted archive; both
used to be bare asserts, so the caller got an ``AssertionError`` with no message
and nothing at all under ``python -O``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from xorq.common.exceptions import XorqInputError
from xorq.common.utils.download_utils import extract_zip


def single_rooted_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("root/file.txt", b"")
    return path


def test_extract_zip_refuses_to_overwrite(tmp_path: Path) -> None:
    source = single_rooted_zip(tmp_path / "archive.zip")
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(FileExistsError, match="target"):
        extract_zip(source, target)


def test_extract_zip_rejects_multi_rooted_archive(tmp_path: Path) -> None:
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("first/file.txt", b"")
        zf.writestr("second/file.txt", b"")
    with pytest.raises(XorqInputError, match="single top-level directory"):
        extract_zip(source, tmp_path / "target")
