import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo


def test_no_name_register():
    xo.connect().register(pd.DataFrame({"a": [1]}))


def test_register_memtable():
    con = xo.connect()
    # memtable: no backends, not unbound → execute() → pa.Table path (not IbisTableProvider)
    t = con.register(xo.memtable({"a": [1, 2, 3]}), table_name="from_memtable")
    assert "from_memtable" in con.list_tables()
    assert t.count().execute() == 3


def test_register_table_provider():
    con = xo.connect()
    con.create_table("src", pa.table({"a": [1, 2, 3]}))
    source = con.table("src")
    # explicit name lands in the catalog
    con.register_table_provider(source, table_name="via_provider")
    assert "via_provider" in con.list_tables()
    # None table_name generates a name; returned table reflects it
    t = con.register_table_provider(source)
    assert t.get_name() in con.list_tables()


def test_register_unbound_expr_raises():
    con = xo.connect()
    t = xo.table({"a": "int64"}, name="unbound")
    with pytest.raises(ValueError, match="unbound tables"):
        con.register(t)
