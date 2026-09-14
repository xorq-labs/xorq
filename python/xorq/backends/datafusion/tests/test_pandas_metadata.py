"""Cross joins of pandas-sourced tables on the datafusion backend (xorq #2266).

See ``xorq/backends/xorq_datafusion/tests/test_pandas_metadata.py`` for the
description of the metadata-sensitive schema equality this guards against.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pytest
from deltalake import write_deltalake

import xorq.api as xo
from xorq.backends.tests.pandas_metadata_util import PANDAS_SOURCES, engine_schema


@pytest.mark.parametrize("to_source", PANDAS_SOURCES)
def test_cross_join_of_pandas_sourced_tables(
    to_source: Callable[[pd.DataFrame], Any],
) -> None:
    con = xo.datafusion.connect()
    con.create_table("a", to_source(pd.DataFrame({"k": ["x"], "v": [1]})))
    con.create_table("b", to_source(pd.DataFrame({"g": ["y"]})))

    actual = (
        con.table("a")
        .cross_join(con.table("b"))
        .group_by("g")
        .aggregate(n=xo._.v.sum())
        .execute()
    )
    assert actual.to_dict("records") == [{"g": "y", "n": 1}]


@pytest.mark.parametrize("to_source", PANDAS_SOURCES)
def test_created_table_has_no_pandas_metadata(
    to_source: Callable[[pd.DataFrame], Any],
) -> None:
    con = xo.datafusion.connect()
    con.create_table("a", to_source(pd.DataFrame({"k": ["x"], "v": [1]})))

    assert engine_schema(con, "a").metadata is None


# The vendored ``_register`` rejects readers outright.
@pytest.mark.parametrize(
    "to_source", [p for p in PANDAS_SOURCES if "record-batch-reader" not in p.id]
)
def test_registered_table_has_no_pandas_metadata(
    to_source: Callable[[pd.DataFrame], Any],
) -> None:
    con = xo.datafusion.connect()
    con.register(to_source(pd.DataFrame({"k": ["x"], "v": [1]})), "a")

    assert engine_schema(con, "a").metadata is None


def test_cross_join_of_memtables() -> None:
    """``_register_in_memory_table`` is the path a memtable reaches."""
    con = xo.datafusion.connect()
    a = xo.memtable({"k": ["x"], "v": [1]})
    b = xo.memtable({"g": ["y"]})

    actual = con.execute(a.cross_join(b).group_by("g").aggregate(n=xo._.v.sum()))
    assert actual.to_dict("records") == [{"g": "y", "n": 1}]
    assert engine_schema(con, a.op().name).metadata is None
    assert engine_schema(con, b.op().name).metadata is None


def test_cross_join_of_registered_against_created_table() -> None:
    """``register`` is a live door that bypasses ``create_table``'s strip."""
    con = xo.datafusion.connect()
    con.register(pa.Table.from_pandas(pd.DataFrame({"k": ["x"], "v": [1]})), "a")
    con.create_table("b", pd.DataFrame({"g": ["y"]}))

    actual = con.execute(
        con.table("a")
        .cross_join(con.table("b"))
        .group_by("g")
        .aggregate(n=xo._.v.sum())
    )
    assert actual.to_dict("records") == [{"g": "y", "n": 1}]
    assert engine_schema(con, "a").metadata is None


def test_cross_join_of_delta_against_created_table(tmp_path) -> None:
    """A Delta table joins a ``create_table`` table.

    ``read_delta`` registers a dataset rather than going through
    ``create_table``; this is a round-trip smoke test, not a
    metadata-discriminating one -- ``to_pyarrow_dataset`` builds its schema
    from the Delta log, so no pandas metadata reaches the register path.
    """
    con = xo.datafusion.connect()
    path = tmp_path / "delta"
    write_deltalake(str(path), pd.DataFrame({"k": ["x"], "v": [1]}))
    a = con.read_delta(path, "a")
    con.create_table("b", pd.DataFrame({"g": ["y"]}))

    actual = con.execute(
        a.cross_join(con.table("b")).group_by("g").aggregate(n=xo._.v.sum())
    )
    assert actual.to_dict("records") == [{"g": "y", "n": 1}]
