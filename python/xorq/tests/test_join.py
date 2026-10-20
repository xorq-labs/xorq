import pandas as pd
import pytest

import xorq.api as xo


def pub_over_join(how):
    # xorq-labs/xorq#2408: `pub` is a coalesce over two join fields, not a
    # plain column, so the chain carries it as a value rather than a field
    a = xo.memtable({"a_pub": [1, -1]})
    b = xo.memtable({"e_pub": [1, 2]})
    j = getattr(a, f"{how}_join")(b, a.a_pub == b.e_pub).select(
        pub=xo._.a_pub.fill_null(xo._.e_pub)
    )
    return j


def sort_pub_pid(df):
    return (
        df[["pub", "pid"]].sort_values("pub").reset_index(drop=True).astype("float64")
    )


def executed_pub_pid(expr):
    return sort_pub_pid(expr.execute())


@pytest.mark.parametrize(
    "how,expected",
    [
        ("outer", {"pub": [-1, 1, 2], "pid": [None, 1, None]}),
        ("inner", {"pub": [1], "pid": [1]}),
    ],
)
@pytest.mark.parametrize(
    "make_predicate",
    [
        pytest.param(lambda j, m: j.pub == m.pid, id="value"),
        pytest.param(lambda j, m: xo._.pub == m.pid, id="deferred"),
        pytest.param(lambda j, m: [("pub", "pid")], id="pair"),
    ],
)
def test_join_on_column_selected_over_join(how, expected, make_predicate):
    j = pub_over_join(how)
    m = xo.memtable({"pid": [1]})
    expr = j.left_join(m, make_predicate(j, m))
    pd.testing.assert_frame_equal(
        executed_pub_pid(expr), pd.DataFrame(expected, dtype="float64")
    )


def test_join_on_column_selected_over_join_after_mutate():
    # the mutate puts a projection between the select and the join; this
    # shape never failed and pins the baseline behaviour
    j = pub_over_join("outer")
    j = j.mutate(flag=j.pub == -1)
    m = xo.memtable({"pid": [1]})
    expr = j.left_join(m, j.pub == m.pid)
    pd.testing.assert_frame_equal(
        executed_pub_pid(expr),
        pd.DataFrame({"pub": [-1, 1, 2], "pid": [None, 1, None]}, dtype="float64"),
    )


def test_asof_join_on_column_selected_over_join():
    # executed on duckdb: the default backend rejects `ASOF LEFT JOIN` syntax
    # for any asof over a join chain, which is unrelated to how the predicate
    # is dereferenced
    j = pub_over_join("outer")
    m = xo.memtable({"pid": [1]})
    expr = j.asof_join(m, on=j.pub >= m.pid)
    pd.testing.assert_frame_equal(
        sort_pub_pid(xo.duckdb.connect().execute(expr)),
        pd.DataFrame({"pub": [-1, 1, 2], "pid": [None, 1, 1]}, dtype="float64"),
    )


def test_join_chain_to_its_own_filtered_result():
    # the right table derives from the chain, so its fields backtrack to the
    # chain's fields; the predicate's left side must still bind to the left
    a = xo.memtable({"id": [1, 2], "x": ["p", "q"]})
    b = xo.memtable({"id": [1, 2], "y": [10, 20]})
    j = a.join(b, "id").select("id", "x", "y")
    r = j.filter(j.id == 1)
    expr = j.left_join(r, j.id == r.id)
    df = expr.execute().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df[["id", "id_right"]].astype("float64"),
        pd.DataFrame({"id": [1, 2], "id_right": [1, None]}, dtype="float64"),
    )
