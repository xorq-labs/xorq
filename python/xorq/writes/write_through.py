"""WriteThrough implementations for ``TeeNode`` deferred writes. See ADR-0014."""

from __future__ import annotations

import abc
import inspect
import os
import queue
import sys
import tempfile
import threading
import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from attr import Attribute, field, frozen
from attr.validators import instance_of

from xorq.common.compat import flock_exclusive
from xorq.writes.enums import WriteMode


if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa

    from xorq.vendor.ibis.backends import BaseBackend


def _coerce_write_mode(value: str | WriteMode) -> WriteMode:
    if isinstance(value, WriteMode):
        return value
    try:
        return WriteMode(value)
    except ValueError:
        raise ValueError(
            f"mode must be one of {tuple(WriteMode)}, got {value!r}"
        ) from None


def _has_read_record_batches(
    instance: object, attribute: Attribute, value: object
) -> None:
    if not callable(getattr(value, "read_record_batches", None)):
        raise TypeError(
            f"con must have a read_record_batches method, got {type(value).__name__}"
        )


class WriteThrough(abc.ABC):
    """A side-effect consumer that wraps a batch stream. See ADR-0014."""

    @abc.abstractmethod
    def write_through(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        """Pull from *batches*, write each as a side effect, yield onward."""
        ...


@frozen
class ParquetWriteThrough(WriteThrough):
    """WriteThrough that writes to a single parquet file with atomic publish.

    ``mode="create"`` raises `FileExistsError` if the target exists.
    ``mode="append"`` merges new data with any existing file under a lock.
    Batches are staged to a temp file; an error mid-stream discards it.
    """

    path: Path = field(converter=Path)
    mode: WriteMode = field(default=WriteMode.CREATE, converter=_coerce_write_mode)

    def __dasher_tokenize__(self) -> tuple:
        return ("parquet-write-through", str(self.path), self.mode)

    def write_through(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        import pyarrow.parquet as pq  # noqa: PLC0415

        writer = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_str = tempfile.mkstemp(
            suffix=".tmp", prefix=self.path.name + ".", dir=self.path.parent
        )
        os.close(fd)
        tmp = Path(tmp_str)
        exhausted = False
        try:
            for batch in batches:
                if writer is None:
                    if self.mode is WriteMode.CREATE and self.path.exists():
                        raise FileExistsError(
                            f"create target already exists: {self.path}"
                        )
                    writer = pq.ParquetWriter(str(tmp), batch.schema)
                writer.write_batch(batch)
                yield batch
            exhausted = True
        except BaseException:
            if writer is not None:
                writer.close()
                writer = None
            tmp.unlink(missing_ok=True)
            raise
        if exhausted and writer is not None:
            try:
                writer.close()
                self._publish(tmp)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        else:
            tmp.unlink(missing_ok=True)

    def _publish(self, tmp: Path) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415

        if self.mode is WriteMode.CREATE:
            try:
                os.link(str(tmp), str(self.path))
            except FileExistsError:
                raise FileExistsError(
                    f"create target already exists: {self.path}"
                ) from None
            finally:
                tmp.unlink(missing_ok=True)
        else:
            # The lock file is a permanent sidecar, never unlinked: removing it
            # would let a later appender open a fresh inode while another still
            # holds the old one's flock, breaking mutual exclusion (concurrent
            # merge-then-rename, last rename wins, rows silently dropped).
            lock_path = Path(str(self.path) + ".lock")
            with open(lock_path, "w") as lock_fd:
                flock_exclusive(lock_fd)
                if self.path.exists():
                    merged = Path(str(self.path) + ".merge.tmp")
                    try:
                        existing = pq.ParquetFile(str(self.path))
                        staged = pq.ParquetFile(str(tmp))
                        with pq.ParquetWriter(
                            str(merged), existing.schema_arrow
                        ) as writer:
                            for batch in existing.iter_batches():
                                writer.write_batch(batch)
                            for batch in staged.iter_batches():
                                writer.write_batch(batch)
                        merged.rename(self.path)
                    except BaseException:
                        merged.unlink(missing_ok=True)
                        raise
                    finally:
                        tmp.unlink(missing_ok=True)
                else:
                    tmp.rename(self.path)


@frozen
class BackendWriteThrough(WriteThrough):
    """WriteThrough that writes to a table on any xorq backend.

    Backends that accept ``mode`` are ingested per-batch (each batch commits
    independently — partial writes on mid-stream error). Backends without
    ``mode`` (DataFusion, DuckDB, Pandas) buffer all batches in memory and
    register a single table after the stream is fully consumed.
    """

    con: BaseBackend = field(validator=_has_read_record_batches)
    table_name: str = field(validator=instance_of(str))
    mode: WriteMode = field(default=WriteMode.CREATE, converter=_coerce_write_mode)
    # Not identity-bearing: kwargs tune write mechanics (batch size, compression, etc.),
    # not the logical result, so they are excluded from hash/eq/__dasher_tokenize__.
    kwargs: dict[str, Any] = field(
        factory=dict, hash=False, eq=False, validator=instance_of(dict)
    )

    def __dasher_tokenize__(self) -> tuple:
        return (
            "backend-write-through",
            getattr(self.con, "name", ""),
            self.table_name,
            self.mode,
        )

    @cached_property
    def _supports_mode(self) -> bool:
        sig = inspect.signature(self.con.read_record_batches)
        return "mode" in sig.parameters

    def _ingest(self, reader: pa.RecordBatchReader, mode: str) -> None:
        kw = dict(self.kwargs)
        if self._supports_mode:
            kw["mode"] = mode
        self.con.read_record_batches(reader, table_name=self.table_name, **kw)

    def write_through(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        if self._supports_mode:
            yield from self._write_per_batch(batches)
        else:
            yield from self._write_bulk(batches)

    def _write_per_batch(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        # NOTE: non-atomic — each batch commits independently, so a mid-stream
        # error leaves earlier batches written.
        import pyarrow as pa  # noqa: PLC0415

        first = True
        for batch in batches:
            reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
            if first:
                mode = "create" if self.mode is WriteMode.CREATE else "append"
                first = False
            else:
                mode = "append"
            self._ingest(reader, mode)
            yield batch

    def _write_bulk(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        # NOTE: non-atomic — all batches are yielded downstream BEFORE the
        # backend ingest runs. If _ingest fails the downstream consumer has
        # already received every batch, so the tee "write" guarantee is lost.
        import pyarrow as pa  # noqa: PLC0415

        warnings.warn(
            f"BackendWriteThrough to {getattr(self.con, 'name', '')!r} "
            f"table {self.table_name!r} uses the bulk path: this backend has no "
            "mode-based ingest, so every batch is delivered downstream before "
            "the single post-stream write runs. A write failure therefore "
            "leaves downstream having consumed data that was never persisted, "
            "and the write is not atomic. See ADR-0014.",
            stacklevel=2,
        )
        collected = []
        for batch in batches:
            collected.append(batch)
            yield batch
        if collected:
            reader = pa.RecordBatchReader.from_batches(collected[0].schema, collected)
            self._ingest(reader, self.mode.value)


@frozen
class ThreadedBackendWriteThrough(BackendWriteThrough):
    """BackendWriteThrough that streams a single ingest on a background thread.

    Batches are pushed to a queue and yielded downstream; a background thread
    drains the queue into a single ``read_record_batches`` call.

    ``maxsize`` is the backpressure dial. ``0`` (default) is an unbounded queue:
    the producer never blocks, so a slow write only costs memory — O(lag)
    buffered batches. ``>0`` is a bounded queue: a full queue blocks the
    producer, and because the producer is downstream-primary this back-pressures
    *downstream* too, capping buffered memory at ``maxsize`` batches.

    Bounded mode upholds a **discard-on-death** invariant: if the write thread
    dies mid-stream it keeps draining (discarding) the queue until it sees the
    producer's terminal sentinel, so the producer's blocking ``put`` can never
    wedge on a full queue nobody reads. A dead write also cuts downstream off at
    the failure point — the in-flight batch is not yielded — so a write failure
    is fatal to the read side rather than silently handing out data as though the
    write had succeeded. ``maxsize`` is identity-neutral.
    """

    # Not identity-bearing: transport tuning, not the logical result.
    maxsize: int = field(default=0, hash=False, eq=False, validator=instance_of(int))

    def write_through(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        import pyarrow as pa  # noqa: PLC0415

        q: queue.Queue | queue.SimpleQueue = (
            queue.Queue(self.maxsize) if self.maxsize > 0 else queue.SimpleQueue()
        )
        error: list[BaseException] = []
        sentinel_seen: list[bool] = []

        def write_thread(schema: pa.Schema) -> None:
            def drain() -> Iterator[pa.RecordBatch]:
                while True:
                    item = q.get()
                    # None: clean exhaustion. A BaseException: upstream failed or
                    # the downstream consumer stopped early — end the reader short.
                    if item is None or isinstance(item, BaseException):
                        sentinel_seen.append(True)
                        return
                    yield item

            try:
                reader = pa.RecordBatchReader.from_batches(schema, drain())
                self._ingest(reader, self.mode.value)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
            finally:
                # discard-on-death: if the ingest died before drain() reached the
                # producer's terminal sentinel, keep draining (discarding) so a
                # bounded producer's put() cannot wedge on a full queue nobody
                # reads. Bounded by the sentinel the producer is guaranteed to
                # send, so this cannot loop forever.
                if error and not sentinel_seen:
                    while True:
                        item = q.get()
                        if item is None or isinstance(item, BaseException):
                            break

        thread: threading.Thread | None = None
        try:
            for batch in batches:
                if error:
                    # the write thread already died; stop feeding a queue nobody
                    # drains and surface its error after the join below.
                    break
                if thread is None:
                    thread = threading.Thread(
                        target=write_thread, args=(batch.schema,), daemon=True
                    )
                    thread.start()
                q.put(batch)
                if error:
                    # the write died while this batch was in flight: do not hand
                    # it downstream. A bounded write failure is fatal to the read
                    # side; surface the error after the join below.
                    break
                yield batch
            q.put(None)
        except BaseException as exc:
            q.put(exc)
            raise
        finally:
            if thread is not None:
                thread.join()
            # Surface a write-thread error only if nothing is already
            # propagating; otherwise a late write failure here would mask the
            # in-flight upstream/GeneratorExit error (the root cause), which
            # survives on __context__ but should remain the raised exception.
            if error and sys.exc_info()[0] is None:
                raise error[0]


@frozen
class WritePrimaryWriteThrough(WriteThrough):
    """Run an inner ``WriteThrough`` *write-primary* — the mirror of the default
    downstream-primary transport.

    In the default transport the downstream pull drives the write: the write can
    lag behind (unbounded queue) but never lead, and an early-stopping downstream
    needs ``drain=True`` to finish the write. Here the **write** owns the pull
    loop on a background thread and downstream consumes the already-written
    batches through a single 1:1 queue (no fan-out — write and pass-through are
    the same batch sequence at the same position). The write therefore runs to
    completion independently of the downstream consumption rate: it is
    intrinsically drain-always, so an early stop cannot abort it.

    Transport is identity-neutral: ``__dasher_tokenize__`` delegates to the inner
    writer, so wrapping does not change the cache key or build artifact (the same
    principle as ``drain``).

    ``maxsize`` is the backpressure dial: ``0`` (default) is an unbounded queue
    (write races ahead, downstream may lag in memory); ``>0`` is a bounded queue
    where a full queue blocks the write — i.e. downstream back-pressures the
    write. After an early stop the generator keeps draining (discarding) so a
    bounded write thread cannot deadlock on a full queue nobody reads.
    """

    inner: WriteThrough = field(validator=instance_of(WriteThrough))
    # Not identity-bearing: transport tuning, not the logical result.
    maxsize: int = field(default=0, hash=False, eq=False, validator=instance_of(int))

    def __dasher_tokenize__(self) -> tuple:
        return self.inner.__dasher_tokenize__()

    def write_through(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        done = object()
        q: queue.Queue | queue.SimpleQueue = (
            queue.Queue(self.maxsize) if self.maxsize > 0 else queue.SimpleQueue()
        )
        error: list[BaseException] = []

        def write_thread() -> None:
            try:
                for batch in self.inner.write_through(batches):
                    q.put(batch)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
            finally:
                q.put(done)

        thread = threading.Thread(target=write_thread, daemon=True)
        thread.start()
        draining = False
        try:
            while True:
                item = q.get()
                if item is done:
                    break
                if not draining:
                    try:
                        yield item
                    except GeneratorExit:
                        # downstream stopped early: keep draining the queue so the
                        # write thread runs to completion (write-primary is
                        # drain-always), then fall through to join below.
                        draining = True
        finally:
            thread.join()
            if error:
                raise error[0]


class DrainingIterator:
    """Wraps a write-through generator; drains remaining batches on close.

    Callers must call ``close()`` then ``join()`` — there is no ``__del__``
    finalizer; the execution pipeline (``api.py``) owns the lifecycle.
    """

    def __init__(self, write_gen: Iterator[pa.RecordBatch]) -> None:
        self._gen = write_gen
        self._exhausted = False
        self._drain_thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._exhausted

    def __iter__(self) -> DrainingIterator:  # noqa: PYI034
        return self

    def __next__(self) -> pa.RecordBatch:
        try:
            return next(self._gen)
        except StopIteration:
            with self._lock:
                self._exhausted = True
            raise

    def _drain(self) -> None:
        try:
            for _ in self._gen:
                pass
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
        with self._lock:
            self._exhausted = True

    def close(self) -> None:
        with self._lock:
            if self._exhausted or self._drain_thread is not None:
                return
            self._drain_thread = threading.Thread(target=self._drain)
            self._drain_thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=timeout)
            if self._drain_thread.is_alive():
                raise TimeoutError("drain thread did not finish within timeout")
            if self._error is not None:
                raise self._error
        elif not self.exhausted:
            raise RuntimeError(
                "join() called before close() on a partially-consumed DrainingIterator"
            )
