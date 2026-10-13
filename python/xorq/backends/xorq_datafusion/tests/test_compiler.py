import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.backends.xorq_datafusion.compiler import DataFusionType


@pytest.fixture
def strings():
    return xo.memtable(pd.DataFrame({"s": ["x", "y"]}), name="strings")


def test_large_string_cast_compiles_to_arrow_cast(strings):
    sql = xo.to_sql(strings.s.cast(dt.LargeString()))
    assert 'ARROW_CAST("t0"."s", \'LargeUtf8\')' in sql


def test_large_string_cast_executes_as_large_string(con, strings):
    expr = strings.s.cast(dt.LargeString())
    result = con.to_pyarrow(expr)
    assert result.type == pa.large_string()
    assert result.to_pylist() == ["x", "y"]


@pytest.mark.parametrize(
    ("type_string", "expected"),
    [
        ("utf8", dt.string),
        ("utf8view", dt.string),
        ("float64", dt.float64),
        ("date32", dt.date),
    ],
)
def test_type_mapper_reads_arrow_type_strings(type_string, expected):
    assert DataFusionType.from_string(type_string) == expected
