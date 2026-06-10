"""Deferred write as a side effect: the consumer side of `TeeNode`/`SinkNode`.

See ADR-0014. A `TeeNode` drives a generic RecordBatch consumer: it calls
``consumer.read(batch)`` for each batch as it flows downstream, then
``consumer.commit()`` on exhaustion (or ``consumer.abort()`` on early stop or
error). `ParquetSink` is one such consumer; it owns its own durability,
staging to a temp file and renaming into place only on commit.
"""

import uuid
from pathlib import Path

import pyarrow.parquet as pq


class ParquetSink:
    """A RecordBatch consumer that writes to a directory of parquet files.

    Contract (the tee consumer protocol):

    - ``read(batch)``  — receive a `pyarrow.RecordBatch` (staged to a temp file).
    - ``commit()``     — publish atomically: rename the staged temp into the
      target directory. ``mode="append"`` adds a new file per run;
      ``mode="create"`` first clears the directory's parquet files.
    - ``abort()``      — discard the staged temp; publish nothing.

    Nothing is published until ``commit``, so an early stop or error leaves any
    prior contents intact.
    """

    def __init__(self, path, mode="append"):
        if mode not in ("create", "append"):
            raise ValueError(f"mode must be 'create' or 'append', got {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self._writer = None
        self._tmp = None

    def read(self, batch):
        if self._writer is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._tmp = self.path / f"{uuid.uuid4().hex}.parquet.tmp"
            self._writer = pq.ParquetWriter(str(self._tmp), batch.schema)
        self._writer.write_batch(batch)

    def commit(self):
        if self._writer is None:
            return  # nothing was read: publish nothing
        self._writer.close()
        self._writer = None
        if self.mode == "create":
            for existing in self.path.glob("*.parquet"):
                existing.unlink()
        # same filesystem as the target dir, so the rename is atomic
        self._tmp.rename(self.path / f"{uuid.uuid4().hex}.parquet")
        self._tmp = None

    def abort(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)
            self._tmp = None
