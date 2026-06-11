from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from xorq.common.utils.file_utils import file_digest


def test_file_digest_str_path(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello")
    expected = hashlib.md5(b"hello").hexdigest()
    assert file_digest(str(p)) == expected


def test_file_digest_path_object(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello")
    expected = hashlib.md5(b"hello").hexdigest()
    assert file_digest(p) == expected

    def test_string_path(self, sample_file):
        result = file_digest(str(sample_file))
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

def test_file_digest_cached_returns_same(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello")
    first = file_digest(p)
    second = file_digest(p)
    assert first == second

    def test_zip_ext_file(self, tmp_path):
        zp = tmp_path / "test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("inner.txt", "hello world")
        with zipfile.ZipFile(zp, "r") as zf:
            with zf.open("inner.txt") as entry:
                result = file_digest(entry)
        expected = hashlib.md5(b"hello world").hexdigest()
        assert result == expected

def test_file_digest_unsupported_type_raises():
    obj = io.BytesIO(b"hello")
    if hasattr(hashlib, "file_digest"):
        with pytest.raises(ValueError, match="Don't know how to handle type"):
            file_digest(obj)
    else:
        result = file_digest(obj)
        assert result == hashlib.md5(b"hello").hexdigest()
