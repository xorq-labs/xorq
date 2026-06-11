from __future__ import annotations

import hashlib
import itertools
from contextlib import closing
from pathlib import Path
from typing import IO, Callable
from zipfile import ZipExtFile


def _manual_file_digest(
    path: str | Path | IO[bytes], digest: Callable = hashlib.md5, size: int = 2**20
) -> str:
    fh = path if hasattr(path, "read") else Path(path).open("rb")
    with closing(fh):
        obj = digest()
        for chunk in itertools.takewhile(
            bool, (fh.read(size) for fh in itertools.repeat(fh))
        ):
            obj.update(chunk)
        return obj.hexdigest()


def file_digest(
    path: str | Path | ZipExtFile, digest: Callable = hashlib.md5, size: int = 2**20
) -> str:
    if hasattr(hashlib, "file_digest"):
        if isinstance(path, ZipExtFile):
            return hashlib.file_digest(path, digest).hexdigest()
        if isinstance(path, (str, Path)):
            with Path(path).open("rb") as fh:
                return hashlib.file_digest(fh, digest).hexdigest()
        raise ValueError(f"Don't know how to handle type {type(path)}")
    return _manual_file_digest(path, digest, size=size)


def normalize_read_path_md5sum(path: str | Path) -> tuple[tuple[str, str], ...]:
    return (("content-md5sum", file_digest(path)),)


def normalize_read_path_stat(path: Path) -> tuple[tuple[str, object], ...]:
    stat = path.stat()
    return tuple(
        (attrname, getattr(stat, attrname))
        for attrname in (
            "st_mtime",
            "st_size",
            "st_ino",
        )
    )
