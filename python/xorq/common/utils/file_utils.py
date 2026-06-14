from __future__ import annotations

import functools
import hashlib
import itertools
from contextlib import closing
from pathlib import Path
from typing import Callable
from zipfile import ZipExtFile


def _manual_file_digest(
    path: str | Path, digest: Callable = hashlib.md5, size: int = 2**20
) -> str:
    fh = path if hasattr(path, "read") else Path(path).open("rb")
    with closing(fh):
        obj = digest()
        for chunk in itertools.takewhile(
            bool, (fh.read(size) for fh in itertools.repeat(fh))
        ):
            obj.update(chunk)
        return obj.hexdigest()


@functools.cache
def _cached_file_digest(
    path: str,
    dev: int,
    ino: int,
    mtime_ns: int,
    size: int,
    algorithm: str = "md5",
    chunk_size: int = 2**20,
) -> str:
    digest = getattr(hashlib, algorithm)
    if hasattr(hashlib, "file_digest"):
        with Path(path).open("rb") as fh:
            return hashlib.file_digest(fh, digest).hexdigest()
    return _manual_file_digest(Path(path), digest, size=chunk_size)


def _digest_to_algorithm(digest: Callable) -> str | None:
    algo = digest.__name__.removeprefix("openssl_")
    if hasattr(hashlib, algo):
        return algo
    return None


def file_digest(
    path: str | Path, digest: Callable = hashlib.md5, size: int = 2**20
) -> str:
    if isinstance(path, (str, Path)):
        p = Path(path)
        st = p.stat()
        algo = _digest_to_algorithm(digest)
        if algo is not None:
            return _cached_file_digest(
                str(p.resolve()),
                st.st_dev,
                st.st_ino,
                st.st_mtime_ns,
                st.st_size,
                algo,
                size,
            )
        with p.open("rb") as fh:
            return hashlib.file_digest(fh, digest).hexdigest()
    elif hasattr(hashlib, "file_digest"):
        if isinstance(path, ZipExtFile):
            return hashlib.file_digest(path, digest).hexdigest()
        raise ValueError(f"Don't know how to handle type {type(path)}")
    else:
        return _manual_file_digest(path, digest, size=size)
