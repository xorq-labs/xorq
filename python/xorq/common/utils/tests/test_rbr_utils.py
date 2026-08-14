"""Unit-layer pins for ``excepts_print_exc`` and ``streaming_split_exchange``.

The end-to-end pin of the abort policy through a real Flight exchange is
``test_streaming_split_exchange_flight_failure_aborts`` in
``xorq.flight.tests.test_flight_exchange``, where it joins the flight tests'
xdist group instead of running concurrently with them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
import toolz

from xorq.common.utils.rbr_utils import (
    excepts_print_exc,
    streaming_split_exchange,
)


SPLIT_KEY = "split"


def make_batch(split: int, n_rows: int) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {SPLIT_KEY: [split] * n_rows, "a": list(range(n_rows))}
    )


def make_split_f(fail_on: int) -> Callable:
    """A per-split ``f`` that raises on the ``fail_on`` split."""

    def f(split_reader: pa.RecordBatchReader) -> pa.RecordBatch:
        table = split_reader.read_all()
        (split,) = set(table[SPLIT_KEY].to_pylist())
        if split == fail_on:
            raise ValueError(f"boom on split {split}")
        return pa.RecordBatch.from_pydict({SPLIT_KEY: [split], "n": [table.num_rows]})

    return f


class FakeFlightReader:
    """Duck-types the flight reader `streaming_split_exchange` consumes.

    ``make_filtered_reader`` needs ``.schema`` and iteration yielding chunks
    with a ``.data`` record batch.
    """

    def __init__(self, batches: tuple[pa.RecordBatch, ...]) -> None:
        self.schema: pa.Schema = batches[0].schema
        self._chunks = tuple(SimpleNamespace(data=batch) for batch in batches)

    def __iter__(self) -> Iterator[SimpleNamespace]:
        return iter(self._chunks)


class RecordingWriter:
    """Duck-types the flight writer, recording what reached the wire."""

    def __init__(self) -> None:
        self.schema: pa.Schema | None = None
        self.batches: list[pa.RecordBatch] = []

    def begin(self, schema: pa.Schema, options: Any = None) -> None:
        self.schema = schema

    def write_batch(self, batch: pa.RecordBatch) -> None:
        self.batches.append(batch)


def test_excepts_print_exc_default_reraises(
    capsys: pytest.CaptureFixture,
) -> None:
    """The default handler prints the traceback, then propagates.

    Both halves matter: printing server-side is a feature (#1277), and
    swallowing turned failures into empty results (#2227).
    """

    def boom() -> None:
        raise ValueError("boom-message")

    wrapped = excepts_print_exc(boom)
    with pytest.raises(ValueError, match="boom-message"):
        wrapped()
    captured = capsys.readouterr()
    assert "Traceback" in captured.err
    assert "boom-message" in captured.err


def test_excepts_print_exc_swallowing_is_opt_in(
    capsys: pytest.CaptureFixture,
) -> None:
    """Returning ``None`` on failure now requires an explicit handler."""

    def boom() -> None:
        raise ValueError("boom-message")

    wrapped = excepts_print_exc(boom, handler=toolz.functoolz.return_none)
    assert wrapped() is None
    assert "boom-message" in capsys.readouterr().err


def test_streaming_split_exchange_propagates_failure(
    capsys: pytest.CaptureFixture,
) -> None:
    """A failing split aborts the exchange instead of yielding an empty stream.

    The traceback is printed exactly once: only the decorator wraps ``f``'s
    exceptions, there is no second wrap inside the loop.
    """
    reader = FakeFlightReader((make_batch(0, 3), make_batch(1, 2)))
    writer = RecordingWriter()
    with pytest.raises(ValueError, match="boom on split 0"):
        streaming_split_exchange(SPLIT_KEY, make_split_f(0), None, reader, writer)
    assert writer.schema is None
    assert writer.batches == []
    assert capsys.readouterr().err.count("Traceback (most recent call last)") == 1


def test_streaming_split_exchange_delivers_splits_before_failure() -> None:
    """Abort is not atomic: splits before the failing one are already written.

    This pins the documented caveat -- a client that persists batches
    incrementally can retain partial data.
    """
    reader = FakeFlightReader((make_batch(0, 3), make_batch(1, 2), make_batch(2, 1)))
    writer = RecordingWriter()
    with pytest.raises(ValueError, match="boom on split 1"):
        streaming_split_exchange(SPLIT_KEY, make_split_f(1), None, reader, writer)
    assert writer.schema is not None
    assert [batch[SPLIT_KEY].to_pylist() for batch in writer.batches] == [[0]]
