import pandas as pd
import pytest
from pytest import param

import xorq as xo
from xorq.tests.util import assert_frame_equal


def test_simple_agg_ops_read_parquet(data_dir):
    path = data_dir / "parquet" / "functional_alltypes.parquet"

    key_big_int_col = "bigint_col"
    con = xo.connect()
    t = con.read_parquet(path)
    result = (
        t.group_by(key_big_int_col)
        .aggregate(mean=t.double_col.mean(), max=t.timestamp_col.max())
        .execute()
    )
    assert result is not None


def test_simple_agg_ops_read_csv(data_dir):
    path = data_dir / "csv" / "functional_alltypes.csv"

    key_big_int_col = "bigint_col"
    con = xo.connect()
    t = con.read_csv(path)
    result = (
        t.group_by(key_big_int_col)
        .aggregate(mean=t.double_col.mean(), max=t.timestamp_col.max())
        .execute()
    )
    assert result is not None


def test_memtable_ops_dict():
    t = xo.memtable({"s": ["aaa", "a", "aa"]})
    assert t.s.length().execute().gt(0).all()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(
            lambda: xo.memtable([(1, 2.0, "3")], columns=list("abc")),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")),
            id="simple",
        ),
        param(
            lambda: xo.memtable([(1, 2.0, "3")]),
            pd.DataFrame([(1, 2.0, "3")], columns=["col0", "col1", "col2"]),
            id="simple_auto_named",
        ),
        param(
            lambda: xo.memtable(
                pd.DataFrame({"a": [1], "b": [2.0], "c": ["3"]}).astype(
                    {"a": "int8", "b": "float32"}
                )
            ),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="dataframe",
        ),
        param(
            lambda: xo.memtable([dict(a=1), dict(a=2)]),
            pd.DataFrame({"a": [1, 2]}),
            id="list_of_dicts",
        ),
    ],
)
def test_in_memory_table(expr, expected):
    result = expr().execute()
    assert_frame_equal(result, expected)
