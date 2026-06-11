from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from xorq.common.utils.file_utils import (
    _manual_file_digest,
    file_digest,
    normalize_read_path_md5sum,
    normalize_read_path_stat,
)


CONTENT = b"hello world"
CONTENT_MD5 = hashlib.md5(CONTENT).hexdigest()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_bytes(CONTENT)
    return p


def test_file_digest_path_object(sample_file: Path) -> None:
    assert file_digest(sample_file) == CONTENT_MD5


def test_file_digest_string_path(sample_file: Path) -> None:
    assert file_digest(str(sample_file)) == CONTENT_MD5


def test_file_digest_custom_digest(sample_file: Path) -> None:
    result = file_digest(sample_file, digest=hashlib.sha256)
    expected = hashlib.sha256(CONTENT).hexdigest()
    assert result == expected


def test_file_digest_zip_ext_file(tmp_path: Path) -> None:
    zp = tmp_path / "test.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("inner.txt", "hello world")
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open("inner.txt") as entry:
            result = file_digest(entry)
    assert result == CONTENT_MD5


def test_file_digest_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Don't know how to handle type"):
        file_digest(42)


def test_manual_file_digest_with_path(sample_file: Path) -> None:
    assert _manual_file_digest(sample_file) == CONTENT_MD5


def test_manual_file_digest_with_file_like_object() -> None:
    fh = io.BytesIO(CONTENT)
    assert _manual_file_digest(fh) == CONTENT_MD5


def test_normalize_read_path_md5sum(sample_file: Path) -> None:
    result = normalize_read_path_md5sum(sample_file)
    assert result == (("content-md5sum", CONTENT_MD5),)


def test_normalize_read_path_stat(sample_file: Path) -> None:
    result = normalize_read_path_stat(sample_file)
    stat = sample_file.stat()
    assert result == (
        ("st_mtime", stat.st_mtime),
        ("st_size", stat.st_size),
        ("st_ino", stat.st_ino),
    )


def test_normalize_read_path_stat_changes_after_write(sample_file: Path) -> None:
    result_before = normalize_read_path_stat(sample_file)
    sample_file.write_bytes(b"different content")
    result_after = normalize_read_path_stat(sample_file)
    assert result_before != result_after
