import pandas as pd
import pytest

import xorq.api as xo


def test_table_sql():
    con = xo.datafusion.connect()
    t = con.create_table("base", pd.DataFrame({"a": [1, 2, 3]}))

    expr = t.sql("SELECT a FROM base WHERE a > 1")

    assert expr.execute()["a"].tolist() == [2, 3]


def test_con_sql():
    con = xo.datafusion.connect()
    con.create_table("base", pd.DataFrame({"a": [1, 2, 3]}))

    assert con.sql("SELECT sum(a) AS s FROM base").execute()["s"].tolist() == [6]


def test_sql_over_into_backend_table():
    rt = xo.memtable({"a": [1, 2]}).into_backend(xo.datafusion.connect(), "rt")

    assert sorted(rt.sql("SELECT a FROM rt").execute()["a"]) == [1, 2]


def test_sql_schema_probe_leaves_no_view():
    con = xo.datafusion.connect()
    con.create_table("base", pd.DataFrame({"a": [1]}))

    con.sql("SELECT a FROM base")

    assert con.list_tables() == ["base"]


def test_sql_schema_probe_does_not_run_ddl():
    con = xo.datafusion.connect()
    con.create_table("base", pd.DataFrame({"a": [1]}))

    with pytest.raises(Exception, match="DDL not supported"):
        con.sql("DROP TABLE base")

    assert con.list_tables() == ["base"]
