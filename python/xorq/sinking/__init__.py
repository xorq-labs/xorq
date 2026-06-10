"""Deferred write as a side effect: the consumer side of `TeeNode`/`SinkNode`.

See ADR-0014. A `TeeNode` drives a generic RecordBatch consumer: it calls
``consumer.read(batch)`` for each batch as it flows downstream, then
``consumer.commit()`` on exhaustion (or ``consumer.abort()`` on early stop or
error). `ParquetSink` is one such consumer; it owns its own durability,
staging to a temp file and renaming into place only on commit.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class SinkMode(StrEnum):
    CREATE = "create"
    APPEND = "append"


class ParquetSink:
    """A RecordBatch consumer that writes to a directory of parquet files.

    Contract (the tee consumer protocol):

    - ``read(batch)``  — receive a `pyarrow.RecordBatch` (staged to a temp file).
      ``mode="create"`` raises `FileExistsError` here if the target already has
      parquet files (it refuses to overwrite, like SQL ``CREATE TABLE``). Raised
      mid-pull, an engine may surface it wrapped (e.g. as a ``ValueError`` across
      the Arrow C-stream boundary).
    - ``commit()``     — publish atomically: rename the staged temp into the
      target directory as a new file.
    - ``abort()``      — discard the staged temp; publish nothing.

    Nothing is published until ``commit``, so an early stop or error leaves any
    prior contents intact.
    """

    path: Path
    mode: SinkMode
    _writer: pq.ParquetWriter | None
    _tmp: Path | None

    def __init__(
        self, path: str | Path, mode: SinkMode | str = SinkMode.APPEND
    ) -> None:
        """Configure the sink target and write mode.

        Parameters
        ----------
        path
            Target directory for the parquet files.
        mode
            `SinkMode` or one of its values: ``"append"`` adds a file per run,
            ``"create"`` refuses to overwrite an existing target.

        Raises
        ------
        ValueError
            If `mode` is not a `SinkMode` value (re-raised from
            ``SinkMode(mode)``).
        """
        try:
            self.mode = SinkMode(mode)
        except ValueError:
            raise ValueError(
                f"mode must be one of {tuple(SinkMode)}, got {mode!r}"
            ) from None
        self.path = Path(path)
        self._writer = None
        self._tmp = None

    def read(self, batch: pa.RecordBatch) -> None:
        """Receive one batch, staging it to a temp parquet file.

        The first call opens the writer (creating the target directory) and, in
        ``create`` mode, refuses to overwrite an existing target.

        Parameters
        ----------
        batch
            The record batch to stage. The first batch's schema fixes the
            writer's schema; a later batch with a different schema raises from
            pyarrow.

        Raises
        ------
        FileExistsError
            In ``create`` mode, if the target already holds ``*.parquet`` files
            (checked on the first call). Raised mid-pull through an engine it
            may surface wrapped (e.g. as `ValueError` across the Arrow C-stream
            boundary).
        OSError
            From creating the directory or opening/writing the temp file
            (permission denied, disk full); not caught.
        """
        if self._writer is None:
            self.path.mkdir(parents=True, exist_ok=True)
            if self.mode is SinkMode.CREATE and any(self.path.glob("*.parquet")):
                raise FileExistsError(
                    f"create sink target already has parquet files: {self.path}"
                )
            self._tmp = self.path / f"{uuid.uuid4().hex}.parquet.tmp"
            self._writer = pq.ParquetWriter(str(self._tmp), batch.schema)
        self._writer.write_batch(batch)

    def commit(self) -> None:
        """Publish the staged write atomically, or do nothing if nothing was read.

        Closes the writer and renames the temp file into the target directory
        as a new parquet file. Staging happens inside the target directory, so
        the rename is on one filesystem and atomic. A no-op when no batch was
        read (no writer open).

        Raises
        ------
        OSError
            From flushing/closing the writer (disk full) or renaming the temp
            file; not caught.
        """
        if self._writer is None:
            return  # nothing was read: publish nothing
        self._writer.close()
        self._writer = None
        # same filesystem as the target dir, so the rename is atomic
        self._tmp.rename(self.path / f"{uuid.uuid4().hex}.parquet")
        self._tmp = None

    def abort(self) -> None:
        """Discard the staged write; publish nothing. Idempotent.

        Closes the writer if open and unlinks the temp file with
        ``missing_ok=True``, so calling it more than once, or after `commit`,
        is safe.

        Raises
        ------
        OSError
            From closing the writer, in rare cases; not caught.
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)
            self._tmp = None
