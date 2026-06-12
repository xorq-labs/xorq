"""Deferred write as a side effect: the consumer side of `TeeNode`.

See ADR-0014. A `TeeNode` wraps an expression with a `SinkNode` whose
``execute(batches)`` generator pulls from the upstream, writes each batch as a
side effect, and yields it downstream unchanged. `ParquetSink` writes to a
single parquet file; `BackendSink` delegates to a backend's
``read_record_batches`` for per-batch ingest (e.g. Postgres via ADBC).
"""

from __future__ import annotations

import abc
import fcntl
import inspect
import os
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from attr import field, frozen
from attr.validators import instance_of


if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa


class SinkMode(StrEnum):
    CREATE = "create"
    APPEND = "append"


def _coerce_sink_mode(value: str | SinkMode) -> SinkMode:
    if isinstance(value, SinkMode):
        return value
    try:
        return SinkMode(value)
    except ValueError:
        raise ValueError(
            f"mode must be one of {tuple(SinkMode)}, got {value!r}"
        ) from None


class SinkNode(abc.ABC):
    """A side-effect consumer that wraps a batch stream.

    ``execute(batches)`` is a generator: it pulls from *batches*, writes each
    batch as a side effect, and yields it onward. The downstream consumer
    drives iteration — the sink never independently pulls.
    """

    @abc.abstractmethod
    def execute(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Pull from *batches*, write each as a side effect, yield onward."""
        ...


@frozen
class ParquetSink(SinkNode):
    """A SinkNode that writes to a single parquet file.

    Each ``execute`` run stages batches to a temp file and atomically publishes
    on clean exhaustion. An error mid-stream discards the temp file.

    ``mode="create"`` raises `FileExistsError` if the target already exists
    (like SQL ``CREATE TABLE``).
    ``mode="append"`` merges new data with any existing file content under a
    file lock, then atomically swaps the merged file into place.
    """

    path: Path = field(converter=Path)
    mode: SinkMode = field(default=SinkMode.APPEND, converter=_coerce_sink_mode)

    def execute(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        import pyarrow.parquet as pq  # noqa: PLC0415

        writer = None
        tmp = Path(str(self.path) + ".tmp")
        exhausted = False
        try:
            for batch in batches:
                if writer is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
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
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
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
class BackendSink(SinkNode):
    """A SinkNode that writes to a table on any xorq backend.

    Backends that accept a ``mode`` parameter (e.g. Postgres via ADBC) are
    ingested per-batch with create→append mode switching — each batch commits
    immediately, so failures mid-stream leave earlier batches written.

    Backends without ``mode`` (DataFusion, DuckDB, Pandas) register a single
    table from all batches after the stream is fully consumed.  Batches are
    buffered in memory so the registration is a single call.
    """

    con: Any = field()
    table_name: str = field(validator=instance_of(str))
    mode: SinkMode = field(default=SinkMode.CREATE, converter=_coerce_sink_mode)
    kwargs: dict = field(factory=dict, hash=False, eq=False)

    @cached_property
    def _supports_mode(self) -> bool:
        sig = inspect.signature(self.con.read_record_batches)
        return "mode" in sig.parameters

    def _ingest(self, reader: pa.RecordBatchReader, mode: str) -> None:
        kw = dict(self.kwargs)
        if self._supports_mode:
            kw["mode"] = mode
        self.con.read_record_batches(reader, table_name=self.table_name, **kw)

    def execute(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        if self._supports_mode:
            yield from self._execute_per_batch(batches)
        else:
            yield from self._execute_bulk(batches)

    def _execute_per_batch(
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

    def _execute_bulk(
        self, batches: Iterable[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        import pyarrow as pa  # noqa: PLC0415

        collected = []
        for batch in batches:
            collected.append(batch)
            yield batch
        if collected:
            reader = pa.RecordBatchReader.from_batches(collected[0].schema, collected)
            self._ingest(reader, self.mode.value)
