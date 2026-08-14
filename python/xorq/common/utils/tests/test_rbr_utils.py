from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
import toolz

import xorq.api as xo
from xorq.common.utils.rbr_utils import (
    excepts_print_exc,
    streaming_split_exchange,
)
from xorq.flight import FlightServer
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import AbstractExchanger


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


def test_streaming_split_exchange_propagates_failure() -> None:
    """A failing split aborts the exchange instead of yielding an empty stream."""
    reader = FakeFlightReader((make_batch(0, 3), make_batch(1, 2)))
    writer = RecordingWriter()
    with pytest.raises(ValueError, match="boom on split 0"):
        streaming_split_exchange(SPLIT_KEY, make_split_f(0), None, reader, writer)
    assert writer.schema is None
    assert writer.batches == []


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


class FailingSplitExchanger(AbstractExchanger):
    """`streaming_split_exchange` exchanger whose ``f`` raises on split 1."""

    @property
    def exchange_f(self) -> Callable:
        return functools.partial(streaming_split_exchange, SPLIT_KEY, make_split_f(1))

    @property
    def schema_in_required(self) -> None:
        return None

    @property
    def schema_in_condition(self) -> Callable:
        def condition(schema_in: Any) -> bool:
            return any(name == SPLIT_KEY for name in schema_in)

        return condition

    @property
    def calc_schema_out(self) -> Callable:
        def f(schema_in: Any) -> Any:
            return xo.schema({SPLIT_KEY: "int64", "n": "int64"})

        return f

    @property
    def description(self) -> str:
        return "raises on split 1"

    @property
    def command(self) -> str:
        return "failing-split-exchange"

    @property
    def query_result(self) -> dict:
        return {
            "schema-in-required": self.schema_in_required,
            "schema-in-condition": self.schema_in_condition,
            "calc-schema-out": self.calc_schema_out,
            "description": self.description,
            "command": self.command,
        }


def test_streaming_split_exchange_flight_failure_aborts() -> None:
    """End-to-end: a failing split surfaces as an error on the client's reader.

    Runs on a daemon thread with a hard deadline (the ``execute_with_deadline``
    pattern from ``test_flight_exchange.py``) so a regression to swallowing or
    deadlocking fails instead of wedging CI. Also pins the non-atomicity
    caveat end-to-end: the split-0 batch is delivered before the raise.
    """
    batches = (make_batch(0, 3), make_batch(1, 2), make_batch(2, 1))
    rbr_in = pa.RecordBatchReader.from_batches(batches[0].schema, iter(batches))
    exchanger = FailingSplitExchanger()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            with FlightServer() as server:
                client = server.client
                client.do_action(
                    AddExchangeAction.name, exchanger, options=client._options
                )
                (_, rbr_out) = client.do_exchange_batches(exchanger.command, rbr_in)
                delivered: list[pa.RecordBatch] = []
                try:
                    for batch in rbr_out:
                        delivered.append(batch)
                except BaseException as e:  # noqa: BLE001
                    box["error"] = e
                finally:
                    box["delivered"] = delivered
        except BaseException as e:  # noqa: BLE001
            box["setup_error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(90)
    assert not thread.is_alive(), "exchange did not finish within 90s: deadlocked"
    assert "setup_error" not in box, box.get("setup_error")
    assert "error" in box, "failing split produced a clean stream instead of an error"
    assert "boom on split 1" in str(box["error"])
    assert [batch[SPLIT_KEY].to_pylist() for batch in box["delivered"]] == [[0]]
