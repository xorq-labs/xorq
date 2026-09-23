from __future__ import annotations

import datetime
import os
import re
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from attr import field, frozen
from attr.validators import instance_of, optional

from xorq.common.exceptions import CacheIntegrityError


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr.operations.core import Node
    from xorq.vendor.ibis.expr.schema import Schema


def _lazy_backend_validator(instance, attribute, value):
    from xorq.vendor.ibis.backends import BaseBackend  # noqa: PLC0415

    if not isinstance(value, BaseBackend):
        raise TypeError(
            f"'{attribute.name}' must be a BaseBackend "
            f"(got {value!r} that is a {type(value)!r})."
        )


def _lazy_default_backend():
    from xorq.config import default_backend  # noqa: PLC0415

    return default_backend()


def _lazy_default_relative_path():
    from xorq.config import options  # noqa: PLC0415

    return options.get("cache.default_relative_path")


def _convert_optional_path(value):
    return None if value is None else Path(value)


def resolve_parquet_cache_dir(
    relative_path: Path | str,
    base_path: Path | None = None,
) -> Path:
    """Return the directory that holds parquet cache files for the given storage params."""
    if base_path is None:
        from xorq.common.utils.caching_utils import get_xorq_cache_dir  # noqa: PLC0415

        base_path = get_xorq_cache_dir()
    return base_path / relative_path


def resolve_parquet_cache_path(
    relative_path: Path | str,
    key: str,
    base_path: Path | None = None,
) -> Path:
    """Return the full path of the parquet file for *key* under the given storage params."""
    return resolve_parquet_cache_dir(relative_path, base_path) / (key + ".parquet")


@frozen
class CacheStorage:
    @abstractmethod
    def exists(self, key):
        pass

    def is_present(self, key):
        """Whether anything occupies *key*, sound or not.

        `exists` answers "is there a hit here to serve"; this answers "is there
        anything here to drop". They part company over an artifact `exists`
        refuses -- corrupt, or past its TTL -- which still has to be droppable.
        Storages that cannot hold such an artifact need no override.
        """
        return self.exists(key)

    def check_integrity(self, key):
        """Raise CacheIntegrityError if the artifact at *key* cannot be read.

        Silent for a key that holds nothing: absence is a miss, not damage.
        Storages with no notion of a damaged artifact need no override.
        """

    @abstractmethod
    def get(self, key: str, schema: Schema | None = None) -> Node:
        pass

    @abstractmethod
    def put(self, key, value, parquet_metadata=None):
        pass

    @abstractmethod
    def drop(self, key):
        pass


def _write_parquet(path, batch_reader, parquet_metadata=None):
    """Stream *batch_reader* to *path*; return the number of rows written.

    The row count is the caller's only independent handle on what actually
    landed: ``pq.ParquetWriter.__exit__`` calls ``close()`` unconditionally, so
    a footer is finalized even when the batch loop raises, and the resulting
    file looks structurally plausible on its own.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    schema = batch_reader.schema
    if parquet_metadata is not None:
        from xorq.common.utils.provenance_utils import (  # noqa: PLC0415
            inject_metadata_into_schema,
        )

        schema = inject_metadata_into_schema(schema, parquet_metadata)
    n_rows = 0
    with pq.ParquetWriter(str(path), schema) as writer:
        for batch in batch_reader:
            writer.write_batch(batch)
            n_rows += batch.num_rows
    return n_rows


# Verification must not need more memory than the write it checks.
# `iter_batches` defaults to 65536 rows over every column, far wider than the
# batches a backend streams, so a blob-heavy table that wrote in ~8k-row
# batches could fail its own read-back.
_VERIFY_BATCH_SIZE = 8192


def _reraise_read_failure(e, message):
    """Re-raise *e* as CacheIntegrityError, unless it is an OS-level failure.

    pyarrow reports both through the same channel. A corrupt artifact raises an
    ArrowException ("Parquet magic bytes not found in footer") or a bare,
    errno-less OSError ("Invalid column metadata (corrupt file?)"). A real OS
    failure -- EMFILE, a stale NFS handle, a permission change -- raises an
    OSError subclass carrying an errno, and says nothing about the artifact.
    Only the first kind may be called corruption: callers turn corruption into
    a cache miss, and a miss recomputes the expression and then overwrites a
    file that was never bad.
    """
    import pyarrow as pa  # noqa: PLC0415

    is_corrupt = isinstance(e, pa.ArrowException) or (
        type(e) is OSError and e.errno is None
    )
    if not is_corrupt:
        raise e
    raise CacheIntegrityError(f"{message}: {type(e).__name__}: {e}") from e


def read_parquet_metadata(source, name=None):
    """Return *source*'s parquet metadata; raise CacheIntegrityError if unreadable.

    *source* is a path or an open binary file -- remote storages hand us the
    latter -- and *name* is what an error message should call it.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    try:
        return pq.ParquetFile(source).metadata
    except Exception as e:
        _reraise_read_failure(e, f"cache artifact {name or source} could not be read")


def verify_parquet(source, expected_rows, name=None):
    """Raise unless every row group of *source* decodes and the count matches.

    This is a full read, not a footer peek, and deliberately so. The corruption
    this guards against presents as an intact footer over an undecodable body
    -- the footer reports the right row count while the pages beneath it fail
    with "Couldn't deserialize thrift" or "Unexpected end of stream". Only
    decoding the pages distinguishes that from a good file, and a cache write
    has already paid for a full upstream scan by the time we get here.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    name = source if name is None else name
    try:
        pf = pq.ParquetFile(source)
        claimed = pf.metadata.num_rows
        read = sum(
            batch.num_rows for batch in pf.iter_batches(batch_size=_VERIFY_BATCH_SIZE)
        )
    except Exception as e:
        _reraise_read_failure(e, f"cache write to {name} could not be read back")
    if not (claimed == read == expected_rows):
        raise CacheIntegrityError(
            f"cache write to {name} is inconsistent: streamed {expected_rows} rows, "
            f"footer claims {claimed}, {read} readable"
        )


def verify_writes_enabled():
    """Whether to read a cache write back before publishing it.

    On by default: an unverified write is how the reported corruption reached
    disk in the first place. It costs a second full read of what was just
    written, so a deployment that would rather carry the risk than pay the read
    can turn it off with ``options.cache.verify_writes = False``
    (``XORQ_CACHE_VERIFY_WRITES=False``).
    """
    from xorq.config import options  # noqa: PLC0415

    return options.get("cache.verify_writes")


def quarantine(error, tmp_path, move):
    """Restate *error* with where the write it rejected was left.

    The error names the file it rejected, so that file has to still be there
    when someone goes looking -- this is the one failure whose artifact is
    worth reading. ``.corrupt`` puts it out of reach of both the cache
    (``<key>.parquet``) and the reaper. *move* is best-effort: failing to set
    the evidence aside must not replace the error that found it.
    """
    target = f"{str(tmp_path)[: -len('.tmp')]}.corrupt"
    try:
        move(tmp_path, target)
    except Exception:  # noqa: BLE001 - preserving evidence is best-effort
        return CacheIntegrityError(f"{error}; artifact left at {tmp_path}")
    return CacheIntegrityError(f"{error}; artifact preserved at {target}")


# `<key>.parquet.<pid>.<uuid4 hex>.tmp`, the shape `_tmp_path_for` builds.
_tmp_name_pattern = re.compile(r"\.parquet\.\d+\.[0-9a-f]{32}\.tmp\Z")
_reaped_dirs = set()


def _tmp_path_for(path):
    """Return a temp path for *path* that is unique to this writer.

    The temp name must be unique per writer, not just per key. Two processes
    caching the same expression derive the same key and so, on a key-only temp
    name, open and truncate the SAME file: their writes interleave, the later
    finisher renames its footer over the mixed body and returns successfully,
    and the earlier one's rename fails because its temp file has been renamed
    away. The survivor is a valid footer over an undecodable body -- which
    `exists` then reports as a cache hit forever. Reproduced 5 times in 8 runs
    before this change.

    The shape is also what the reaper matches on, so this and
    `_tmp_name_pattern` have to stay in step.
    """
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _reap_stale_tmp(directory, max_age=datetime.timedelta(days=1)):
    """Delete abandoned cache writes of ours in *directory* older than *max_age*.

    A failed write leaves its temp file behind by design -- the rename is
    skipped, so nothing corrupt is ever published -- but nothing has ever
    removed them, and they are full-size. The customer report that prompted
    this carried a 359,893,078-byte orphan that had sat for hours.

    Matched on our own temp shape rather than on ``*.tmp``: a cache directory
    is an ordinary directory, and other writers stage files in it --
    ``ParquetWriteThrough`` puts ``<name>.parquet.<random>.tmp`` beside its
    target -- which are none of our business to delete.

    Age-gated rather than unconditional: a temp file younger than max_age may
    belong to a write that is still streaming. Best-effort throughout; reaping
    must never be the reason a cache write fails.
    """
    cutoff = datetime.datetime.now() - max_age
    try:
        candidates = [
            path
            for path in directory.glob("*.tmp")
            if _tmp_name_pattern.search(path.name)
        ]
    except OSError:
        return
    for stale in candidates:
        try:
            if datetime.datetime.fromtimestamp(stale.stat().st_mtime) < cutoff:
                stale.unlink()
        except OSError:
            continue


def _reap_stale_tmp_once(directory):
    """Reap *directory*, at most once per process.

    Reaping scans the whole cache directory, and `put` is on the miss path of
    every query. A directory holding tens of thousands of artifacts would pay
    that scan on every write, to collect orphans that accumulate over hours.
    """
    if directory in _reaped_dirs:
        return
    _reaped_dirs.add(directory)
    _reap_stale_tmp(directory)


@frozen
class ParquetStorage(CacheStorage):
    source = field(
        validator=_lazy_backend_validator,
        factory=_lazy_default_backend,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=_lazy_default_relative_path,
        converter=Path,
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
        converter=_convert_optional_path,
    )

    def __dasher_tokenize__(self):
        return (
            "normalize_parquet_storage",
            self.source,
            self.relative_path,
            self.base_path,
        )

    def _ensure_dir(self) -> None:
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def path(self) -> Path:
        return resolve_parquet_cache_dir(self.relative_path, self.base_path)

    def get_path(self, key):
        return resolve_parquet_cache_path(self.relative_path, key, self.base_path)

    def is_present(self, key):
        return self.get_path(key).exists()

    def check_integrity(self, key):
        # A published artifact is verified at write time, so this is only a
        # backstop for files written by an older xorq (or damaged since). It
        # opens the footer and nothing more: catching truncation and footer
        # damage cheaply on a path taken for every cache-hit decision. It does
        # NOT prove the pages decode -- verify-on-write is what guarantees
        # that -- so treat silence here as "plausible", not "checked".
        path = self.get_path(key)
        if path.exists():
            read_parquet_metadata(path)

    def exists(self, key):
        if not self.is_present(key):
            return False
        try:
            self.check_integrity(key)
        except CacheIntegrityError:
            return False
        except FileNotFoundError:
            # dropped between the two checks: a miss, not a failure
            return False
        return True

    def get(self, key: str, schema: Schema | None = None) -> Node:
        from xorq.common.utils.defer_utils import deferred_read_parquet  # noqa: PLC0415

        # When the caller already knows the schema (e.g. pinning a CachedNode
        # whose schema is on hand), forward it so deferred_read_parquet does not
        # open the parquet footer just to re-infer a schema we already have.
        # `exists` opens the footer too now, for its integrity backstop, so
        # this saves the second of the two opens rather than the only one.
        op = deferred_read_parquet(
            path=self.get_path(key),
            con=self.source,
            table_name=key,
            schema=schema,
        ).op()
        return op

    def put(self, key: str, value: Node, parquet_metadata: dict | None = None) -> Node:
        # Create the cache dir lazily, at write time only -- constructing or
        # relocating a storage (e.g. re-pointing a pinned read at a new
        # base_path on load) must stay free of filesystem side effects.
        self._ensure_dir()
        path = self.get_path(key)
        _reap_stale_tmp_once(path.parent)
        tmp_path = _tmp_path_for(path)
        try:
            with value.to_expr().to_pyarrow_batches() as batch_reader:
                n_rows = _write_parquet(
                    tmp_path, batch_reader, parquet_metadata=parquet_metadata
                )
            # Verify before publishing, never after. Once the rename lands the
            # artifact is indistinguishable from a good one and every later run
            # is a hit on it.
            if verify_writes_enabled():
                verify_parquet(tmp_path, n_rows)
            tmp_path.rename(path)
        except CacheIntegrityError as e:
            raise quarantine(e, tmp_path, os.rename) from e
        except BaseException:
            # Leave nothing half-written behind. The rename above is the only
            # thing that publishes; anything still at tmp_path is ours alone
            # and safe to remove.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return self.get(key)

    def drop(self, key):
        path = self.get_path(key)
        path.unlink()


@frozen
class ParquetTTLStorage(ParquetStorage):
    ttl = field(
        validator=instance_of(datetime.timedelta), default=datetime.timedelta(days=1)
    )

    def __dasher_tokenize__(self):
        return (
            "normalize_parquet_ttl_storage",
            self.source,
            self.relative_path,
            self.base_path,
            self.ttl,
        )

    def exists(self, key):
        # via super(), so the integrity backstop applies here too: a TTL cache
        # must not serve a corrupt artifact for the rest of its window.
        return super().exists(key) and self.satisfies_ttl(self.get_path(key))

    def satisfies_ttl(self, path):
        delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            path.stat().st_mtime
        )
        return delta < self.ttl


@frozen
class ParquetDummyStorage(ParquetStorage):
    def _ensure_dir(self) -> None:
        pass  # dummy storage never touches the filesystem, even on write


@frozen
class SourceStorage(CacheStorage):
    source = field(
        validator=_lazy_backend_validator,
        factory=_lazy_default_backend,
    )

    def __dasher_tokenize__(self):
        return ("normalize_source_storage", self.source)

    def exists(self, key):
        return key in self.source.tables

    def get(self, key: str, schema: Schema | None = None) -> Node:
        # schema is accepted for interface parity with ParquetStorage; a live
        # backend table already carries its schema, so there is no I/O to skip.
        return self.source.table(key).op()

    def put(self, key, value, parquet_metadata=None):
        def is_remote(value):
            name = value.to_expr()._find_backend().name
            # FIXME: add pyiceberg, trino
            return name in ("postgres", "snowflake")

        def is_single_backend(storage, value):
            from xorq.common.utils.graph_utils import find_all_sources  # noqa: PLC0415

            return (storage.source,) == find_all_sources(value.to_expr())

        if is_remote(value):
            if is_single_backend(self, value):
                from xorq.expr.api import remote_table_scope  # noqa: PLC0415

                # must transform for Read ops: create_table expects a vanilla ibis expr
                # full close is safe here: create_table is an eager server-side
                # CTAS and get(key) below never references the placeholders
                with remote_table_scope(value.to_expr()) as transformed_expr:
                    self.source.create_table(key, transformed_expr)
            else:
                assert hasattr(self.source, "read_record_batches")
                # read_record_batches will create durable table in out-of-core fashion
                # works for snowflake and postgres
                self.source.read_record_batches(
                    value.to_expr().to_pyarrow_batches(),
                    key,
                )
        else:
            self.source.create_table(key, value.to_expr().to_pyarrow())
        return self.get(key)

    def drop(self, key):
        self.source.drop_table(key)
