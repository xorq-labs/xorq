from __future__ import annotations

import pandas as pd
import pytest

import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.util import gen_name


@pytest.fixture
def tmp_table(con):
    name = gen_name("databricks_test_table")
    con.create_table(name, obj=pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    yield name
    con.drop_table(name, force=True)


@pytest.mark.parametrize(
    "to_database",
    [
        pytest.param(lambda catalog, database: (catalog, database), id="tuple"),
        pytest.param(lambda catalog, database: f"{catalog}.{database}", id="dotted"),
    ],
)
def test_table_with_database(con, tmp_table, to_database):
    catalog, database = con.current_catalog, con.current_database
    t = con.table(tmp_table, database=to_database(catalog, database))
    assert t.op().namespace == ops.Namespace(catalog=catalog, database=database)
    assert {name: dtype.name for name, dtype in t.schema().items()} == {
        "a": "Int64",
        "b": "String",
    }
    result = t.order_by("a").execute()
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == ["x", "y", "z"]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param(("t", "default"), {}, id="positional-database"),
        pytest.param(("t",), {"schema": "default"}, id="schema-kwarg"),
    ],
)
def test_table_rejects_schema_argument(args, kwargs):
    backend = pytest.importorskip("xorq.backends.databricks.backend").Backend()
    with pytest.raises(TypeError):
        backend.table(*args, **kwargs)
