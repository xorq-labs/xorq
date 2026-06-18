"""Deferred write as a side effect: the consumer side of `TeeNode`.

See ADR-0014. A `TeeNode` wraps an expression with a `Sink` whose
``sink(batches)`` generator pulls from the upstream, writes each batch as a
side effect, and yields it downstream unchanged. `ParquetSink` writes to a
single parquet file; `BackendSink` delegates to a backend's
``read_record_batches`` for per-batch ingest (e.g. Postgres via ADBC).
"""

from __future__ import annotations

import abc
import inspect
import os
import queue
import tempfile
import threading
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from attr import Attribute, field, frozen
from attr.validators import instance_of

from xorq.common.compat import flock_exclusive
from xorq.sinking.enums import SinkMode


if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa

    from xorq.vendor.ibis.backends import BaseBackend


def _coerce_sink_mode(value: str | SinkMode) -> SinkMode:
    if isinstance(value, SinkMode):
        return value
    try:
        return SinkMode(value)
    except ValueError:
        raise ValueError(
            f"mode must be one of {tuple(SinkMode)}, got {value!r}"
        ) from None


def _has_read_record_batches(
    instance: object, attribute: Attribute, value: object
) -> None:
    if not callable(getattr(value, "read_record_batches", None)):
        raise TypeError(
            f"con must have a read_record_batches method, got {type(value).__name__}"
        )


class Sink(abc.ABC):
    """A side-effect consumer that wraps a batch stream.

    ``sink(batches)`` is a generator: it pulls from *batches*, writes each
    batch as a side effect, and yields it onward. The downstream consumer
    drives iteration — the sink never independently pulls.
    """

    @abc.abstractmethod
    def sink(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Pull from *batches*, write each as a side effect, yield onward."""
        ...


@frozen
class ParquetSink(Sink):
    """A Sink that writes to a single parquet file.

    Each ``sink`` run stages batches to a temp file and atomically publishes
    on clean exhaustion. An error mid-stream discards the temp file.

    ``mode="create"`` raises `FileExistsError` if the target already exists
    (like SQL ``CREATE TABLE``).
    ``mode="append"`` merges new data with any existing file content under a
    file lock, then atomically swaps the merged file into place.
    """

    path: Path = field(converter=Path)
    mode: SinkMode = field(default=SinkMode.APPEND, converter=_coerce_sink_mode)

    def __dasher_tokenize__(self) -> tuple:
        return ("ParquetSink", str(self.path), self.mode)

    def sink(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
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
                    if self.mode is SinkMode.CREATE and self.path.exists():
                        raise FileExistsError(
                            f"create sink target already exists: {self.path}"
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

        if self.mode is SinkMode.CREATE:
            try:
                os.link(str(tmp), str(self.path))
            except FileExistsError:
                raise FileExistsError(
                    f"create sink target already exists: {self.path}"
                ) from None
            finally:
                tmp.unlink(missing_ok=True)
        else:
            lock_path = Path(str(self.path) + ".lock")
            try:
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
            finally:
                lock_path.unlink(missing_ok=True)


@frozen
class BackendSink(Sink):
    """A Sink that writes to a table on any xorq backend.

    Backends that accept a ``mode`` parameter (e.g. Postgres via ADBC) are
    ingested per-batch with create→append mode switching — each batch commits
    immediately, so failures mid-stream leave earlier batches written.

    Backends without ``mode`` (DataFusion, DuckDB, Pandas) register a single
    table from all batches after the stream is fully consumed.  Batches are
    buffered in memory so the registration is a single call.
    """

    con: BaseBackend = field(validator=_has_read_record_batches)
    table_name: str = field(validator=instance_of(str))
    mode: SinkMode = field(default=SinkMode.CREATE, converter=_coerce_sink_mode)
    # Not identity-bearing: kwargs tune write mechanics (batch size, compression, etc.),
    # not the logical result, so they are excluded from hash/eq/__dasher_tokenize__.
    kwargs: dict[str, Any] = field(
        factory=dict, hash=False, eq=False, validator=instance_of(dict)
    )

    def __dasher_tokenize__(self) -> tuple:
        return (
            "BackendSink",
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

    def sink(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        if self._supports_mode:
            yield from self._sink_per_batch(batches)
        else:
            yield from self._sink_bulk(batches)

    def _sink_per_batch(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        import pyarrow as pa  # noqa: PLC0415

        first = True
        for batch in batches:
            reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
            if first:
                mode = "create" if self.mode is SinkMode.CREATE else "append"
                first = False
            else:
                mode = "append"
            self._ingest(reader, mode)
            yield batch

    def _sink_bulk(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        # NOTE: non-atomic — all batches are yielded downstream BEFORE the
        # backend ingest runs.  If _ingest fails the downstream consumer has
        # already received every batch, so the tee "write" guarantee is lost.
        import pyarrow as pa  # noqa: PLC0415

        collected = []
        for batch in batches:
            collected.append(batch)
            yield batch
        if collected:
            reader = pa.RecordBatchReader.from_batches(collected[0].schema, collected)
            self._ingest(reader, self.mode.value)


@frozen
class ThreadedBackendSink(BackendSink):
    """A BackendSink that streams a single ingest through a background thread.

    ``sink`` keeps the same generator shape — pull a batch, yield it onward —
    but the side effect is a single ``read_record_batches`` call running on a
    background thread, fed by a queue. Each batch is pushed onto the queue and
    yielded downstream; the thread's reader drains the queue. This replaces both
    the per-batch round-trips (one call per batch, partial commits) and the bulk
    path's full-memory buffer (whole stream in RAM) with one streaming call.

    There is no create→append switching: a single call uses a single ``mode``.
    The ``mode`` kwarg is only forwarded to backends that accept it (inherited
    ``_ingest`` gate), so no-``mode`` backends (DataFusion, DuckDB, Pandas) work
    too.

    The queue is unbounded (``queue.SimpleQueue``): there is no backpressure, so
    a sink slower than the downstream consumer buffers the lag in memory. This
    is the deadlock-safe choice — a bounded queue would block the producer
    forever if the sink thread died without draining. Rollback on a mid-stream
    error follows the backend's own semantics; this is not guaranteed
    all-or-nothing.
    """

    def sink(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        import pyarrow as pa  # noqa: PLC0415

        q: queue.SimpleQueue = queue.SimpleQueue()
        error: list[BaseException] = []

        def sink_thread(schema: pa.Schema) -> None:
            def drain() -> Iterator[pa.RecordBatch]:
                while True:
                    item = q.get()
                    # None: clean exhaustion. A BaseException: upstream failed or
                    # the downstream consumer stopped early — end the reader short.
                    if item is None or isinstance(item, BaseException):
                        return
                    yield item

            try:
                reader = pa.RecordBatchReader.from_batches(schema, drain())
                self._ingest(reader, self.mode.value)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread: threading.Thread | None = None
        try:
            for batch in batches:
                if error:
                    # the sink thread already died; stop feeding a queue nobody
                    # drains and surface its error after the join below.
                    break
                if thread is None:
                    thread = threading.Thread(
                        target=sink_thread, args=(batch.schema,), daemon=True
                    )
                    thread.start()
                q.put(batch)
                yield batch
            q.put(None)
        except BaseException as exc:
            q.put(exc)
            raise
        finally:
            if thread is not None:
                thread.join()
            if error:
                raise error[0]


class DrainingIterator:
    """Wraps a sink generator; drains remaining batches on close.

    Normal iteration is lock-step pass-through.  When the downstream
    consumer closes without exhausting the stream, a non-daemon background
    thread continues iterating the generator so the sink's write
    side-effect runs to completion.

    Callers must call ``close()`` then ``join()`` after downstream
    execution completes.  There is no ``__del__`` finalizer — the
    execution pipeline (``api.py``) owns the lifecycle explicitly.
    """

    def __init__(self, sink_gen: Iterator[pa.RecordBatch]) -> None:
        self._gen = sink_gen
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
            if self._error is not None:
                raise self._error
        elif not self.exhausted:
            raise RuntimeError(
                "join() called before close() on a partially-consumed DrainingIterator"
            )
