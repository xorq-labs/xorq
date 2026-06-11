from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from xorq.common.utils.file_utils import (
    _manual_file_digest,
    file_digest,
    normalize_read_path_md5sum,
    normalize_read_path_stat,
)


@pytest.fixture
def sample_file(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_bytes(b"hello world")
    return p


class TestFileDigest:
    def test_path_object(self, sample_file):
        result = file_digest(sample_file)
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

    def test_string_path(self, sample_file):
        result = file_digest(str(sample_file))
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

    def test_custom_digest(self, sample_file):
        result = file_digest(sample_file, digest=hashlib.sha256)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected

    def test_zip_ext_file(self, tmp_path):
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("inner.txt", "hello world")
        with zipfile.ZipFile(zp, "r") as zf:
            with zf.open("inner.txt") as entry:
                result = file_digest(entry)
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Don't know how to handle type"):
            file_digest(42)


class TestManualFileDigest:
    def test_with_path(self, sample_file):
        result = _manual_file_digest(sample_file)
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

    def test_with_file_like_object(self):
        fh = io.BytesIO(b"hello world")
        result = _manual_file_digest(fh)
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected


class TestNormalizeReadPathMd5sum:
    def test_returns_content_md5sum_tuple(self, sample_file):
        result = normalize_read_path_md5sum(sample_file)
        expected_hash = hashlib.md5(b"hello world").hexdigest()
        assert result == (("content-md5sum", expected_hash),)


class TestNormalizeReadPathStat:
    def test_returns_stat_tuple(self, sample_file):
        result = normalize_read_path_stat(sample_file)
        stat = sample_file.stat()
        assert result == (
            ("st_mtime", stat.st_mtime),
            ("st_size", stat.st_size),
            ("st_ino", stat.st_ino),
        )

    def test_changes_after_write(self, sample_file):
        result_before = normalize_read_path_stat(sample_file)
        sample_file.write_bytes(b"different content")
        result_after = normalize_read_path_stat(sample_file)
        assert result_before != result_after
