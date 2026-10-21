"""What a client sees when its input does not fit an exchange's schema."""

from __future__ import annotations

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.flight import FlightServer
from xorq.flight.exchanger import make_udxf


def test_mismatched_input_schema_names_the_required_schema() -> None:
    schema = xo.schema({"a": "int64"})
    udxf = make_udxf(lambda df: df, schema, schema, name="identity")
    wrong = pa.table({"b": [1]})

    with FlightServer(exchangers=[udxf]) as server:
        with pytest.raises(ValueError, match="requires") as excinfo:
            server.client.do_exchange_batches(udxf.command, wrong.to_reader())

    assert str(schema) in str(excinfo.value)
