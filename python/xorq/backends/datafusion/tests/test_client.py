import pytest
from pytest import param

import xorq.api as xo
from xorq.vendor import ibis


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"obj": xo.memtable({"a": [1]})}, id="obj"),
        param({"schema": ibis.schema({"a": "int64"})}, id="schema"),
    ],
)
def test_create_table_temp(kwargs):
    con = xo.datafusion.connect()
    t = con.create_table("t", temp=True, **kwargs)

    assert t.schema() == ibis.schema({"a": "int64"})
    assert "t" in con.list_tables()
