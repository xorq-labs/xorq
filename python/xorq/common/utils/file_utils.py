from __future__ import annotations

import functools
import hashlib
import itertools
from contextlib import closing
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable
from zipfile import ZipExtFile


if TYPE_CHECKING:
    from xorq.expr.relations import Read


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
    path: str | Path | ZipExtFile, digest: Callable = hashlib.md5, size: int = 2**20
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
        if hasattr(hashlib, "file_digest"):
            with p.open("rb") as fh:
                return hashlib.file_digest(fh, digest).hexdigest()
        return _manual_file_digest(p, digest, size=size)
    elif hasattr(hashlib, "file_digest"):
        if isinstance(path, ZipExtFile):
            return hashlib.file_digest(path, digest).hexdigest()
        if isinstance(path, (str, Path)):
            with Path(path).open("rb") as fh:
                return hashlib.file_digest(fh, digest).hexdigest()
        raise ValueError(f"Don't know how to handle type {type(path)}")
    else:
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


def normalize_read_source_identity(read: Read) -> tuple[tuple[str, object], ...]:
    """Identity for path-less Read ops (e.g. API-backed sources).

    Unlike the path normalizers (which receive a path and hash file
    content/stat), this receives the Read op itself: identity is the source
    profile's content hash plus the read's declarative kwargs. `table_name`
    is excluded (gen_name'd, unstable across constructions) and the profile
    idx suffix is excluded (session-global, unstable across sessions).

    The two contributions are returned *framed* --
    ``(("parts", ...), ("kwargs", ...))`` -- rather than flat-concatenated.
    Flat concatenation is not injective: a source contributing
    ``(("resource", "things"),)`` via ``read_identity_parts`` with no kwargs
    tokenizes identically to a read contributing no parts but carrying a
    ``resource="things"`` kwarg -- two semantically different reads, one
    identity. Framing keeps the encoding injective. This tuple shape is an
    append-only identity contract.
    """
    import toolz  # noqa: PLC0415

    from xorq.common.utils.dasher import tokenize  # noqa: PLC0415

    profile = getattr(read.source, "_profile", None)
    if profile is None:
        raise ValueError(
            f"Read op {getattr(read, 'name', read)!r} has a source without a "
            "profile; normalize_read_source_identity requires one"
        )
    profile_content_hash = tokenize(toolz.dissoc(profile.as_dict(), "idx"))
    parts = (
        ("profile", profile_content_hash),
        ("method_name", read.method_name),
    )
    # a source may contribute identity of its own (RestBackend folds the
    # per-resource config content hash so editing a resource's declarative
    # config changes identity, ADR-0015, without invalidating siblings);
    # delegation keeps backend-specific knowledge out of this normalizer
    read_identity_parts = getattr(read.source, "read_identity_parts", None)
    if read_identity_parts is not None:
        parts += tuple(read_identity_parts(read))
    return (
        ("parts", parts),
        (
            "kwargs",
            tuple(sorted((k, v) for k, v in read.read_kwargs if k != "table_name")),
        ),
    )
