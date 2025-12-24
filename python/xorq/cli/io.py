"""Arrow IPC streaming I/O utilities for CLI."""

import sys
from typing import BinaryIO, Iterator

import pyarrow as pa
from pyarrow import ipc


def write_arrow_stream(
    batches: Iterator[pa.RecordBatch],
    out: BinaryIO = None,
    batch_size: int | None = None,
) -> None:
    """
    Write Arrow IPC stream to output stream (default: stdout).

    Parameters
    ----------
    batches : Iterator[pa.RecordBatch]
        Iterator of Arrow RecordBatch objects to write
    out : BinaryIO, optional
        Binary output stream. Defaults to sys.stdout.buffer
    batch_size : int, optional
        If specified, controls the size of record batches.
        This parameter is for future use and currently not implemented.

    Examples
    --------
    >>> import pyarrow as pa
    >>> table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> write_arrow_stream(table.to_batches())
    """
    if out is None:
        out = sys.stdout.buffer

    first = next(batches, None)
    if first is None:
        return

    writer = ipc.new_stream(out, first.schema)
    try:
        writer.write_batch(first)
        for batch in batches:
            writer.write_batch(batch)
    finally:
        writer.close()


def read_arrow_stream(inp: BinaryIO = None) -> pa.Table:
    """
    Read Arrow IPC stream from input stream (default: stdin).

    Parameters
    ----------
    inp : BinaryIO, optional
        Binary input stream. Defaults to sys.stdin.buffer

    Returns
    -------
    pa.Table
        Arrow table containing all data from the stream

    Examples
    --------
    >>> table = read_arrow_stream()  # reads from stdin
    >>> print(table.schema)
    """
    if inp is None:
        inp = sys.stdin.buffer

    reader = ipc.open_stream(inp)
    return reader.read_all()
