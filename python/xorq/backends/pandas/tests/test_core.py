from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
import xorq.common.exceptions as com
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.pandas import Backend
from xorq.tests.util import assert_frame_equal


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "plain_int64": list(range(1, 4)),
            "plain_strings": list("abc"),
            "dup_strings": list("dad"),
        }
    )


@pytest.fixture
def core_client(dataframe):
    return Backend().connect({"df": dataframe})


@pytest.fixture
def ibis_table(core_client):
    return core_client.table("df")


def test_execute_parameter_only():
    param = xo.param("value", "int64")
    con = xo.pandas.connect()
    result = con.execute(param, params={param.op(): 42})
    assert result == 42


def test_missing_data_sources():
    t = xo.table([("a", "string")], name="t")
    expr = t.a.length()
    con = xo.pandas.connect()
    with pytest.raises(com.UnboundExpressionError):
        con.execute(expr)


def test_unbound_table_execution():
    t = xo.table([("a", "string")], name="t")
    expr = t.a.length()
    con = xo.pandas.connect({"t": pd.DataFrame({"a": ["a", "ab", "abc"]})})
    result = con.execute(expr)
    assert result.tolist() == [1, 2, 3]


def test_view_executes_as_its_table(ibis_table, dataframe):
    assert_frame_equal(ibis_table.view().execute(), dataframe)


def test_view_filter_and_mutate_execute(ibis_table):
    v = ibis_table.view()
    result = v.filter(v.plain_int64 > 1).mutate(doubled=v.plain_int64 * 2).execute()
    expected = pd.DataFrame(
        {
            "plain_int64": [2, 3],
            "plain_strings": ["b", "c"],
            "dup_strings": ["a", "d"],
            "doubled": [4, 6],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(ops.Sum, id="reduction"),
        pytest.param(ops.Mean, id="reduction-mean"),
        pytest.param(ops.Cast, id="visitor"),
        pytest.param(ops.TimestampTruncate, id="visitor-temporal"),
        pytest.param(ops.StringLength, id="kernel"),
    ],
)
def test_has_operation_reports_what_execute_handles(op):
    assert Backend.has_operation(op)


def test_has_operation_matches_execution(ibis_table):
    # a reduction executes, so the capability report must say so
    assert ibis_table.plain_int64.sum().execute() == 6
    assert Backend.has_operation(ops.Sum)


def test_has_operation_rejects_unhandled_op():
    assert not Backend.has_operation(ops.Levenshtein)
