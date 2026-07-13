from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import xorq.api as xo
import xorq.vendor.ibis as ibis
from xorq.backends import _get_backend_names
from xorq.backends.bigquery import Backend
from xorq.vendor.ibis.backends.bigquery import Backend as IbisBigQueryBackend


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr import types as ir


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


def test_backend_registered() -> None:
    assert "bigquery" in _get_backend_names()


def test_backend_subclasses_vendored() -> None:
    assert issubclass(Backend, IbisBigQueryBackend)


def test_api_exposes_backend() -> None:
    assert xo.bigquery.name == "bigquery"
    assert callable(xo.bigquery.connect)
    assert callable(xo.bigquery.compile)


def test_compile_offline() -> None:
    # compilation needs no live connection or credentials
    con = Backend()
    t = ibis.table({"a": "int64", "b": "string"}, name="t")
    sql = con.compile(t.select(t.a + 1))
    assert "SELECT" in sql
    assert "`a`" in sql


@pytest.mark.bigquery
def test_read_parquet_and_execute(batting: ir.Table) -> None:
    result = batting.filter(batting.yearID == 2015).select("playerID").execute()
    assert len(result) > 0
    assert list(result.columns) == ["playerID"]
