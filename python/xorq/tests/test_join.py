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


@pytest.mark.parametrize(
    "make_predicate",
    [
        pytest.param(lambda j, r: j.pub == r.pub, id="value"),
        pytest.param(lambda j, r: xo._.pub == r.pub, id="deferred"),
        pytest.param(lambda j, r: [("pub", "pub")], id="pair"),
    ],
)
def test_join_chain_to_its_own_filtered_result_on_computed_key(make_predicate):
    # `pub` is a coalesce over the chain and the right table derives from the
    # chain, so both sides of the predicate trace back to the same value; the
    # left side must still bind to the chain, not to the right table
    j = pub_over_join("outer")
    r = j.filter(j.pub == 1)
    expr = j.left_join(r, make_predicate(j, r))
    df = expr.execute().sort_values("pub").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df[["pub", "pub_right"]].astype("float64"),
        pd.DataFrame(
            {"pub": [-1, 1, 2], "pub_right": [None, 1, None]}, dtype="float64"
        ),
    )


def test_join_chain_to_its_own_computed_projection():
    # `r` is itself a join chain over the same tables as `j`; its computed
    # field is the right side and must not be peeled onto the left chain
    a = xo.memtable({"id": [1, 2], "x": ["p", "q"]})
    b = xo.memtable({"id": [1, 2], "y": [10, 20]})
    j = a.join(b, "id")
    r = j.select(next_id=j.id + 1)
    expr = j.left_join(r, j.id == r.next_id)
    df = expr.execute().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df[["id", "next_id"]].astype("float64"),
        pd.DataFrame({"id": [1, 2], "next_id": [None, 2]}, dtype="float64"),
    )


@pytest.mark.parametrize(
    "make_right,make_predicate",
    [
        pytest.param(lambda j: j.view(), lambda j, r: j.id == r.id, id="view-value"),
        pytest.param(lambda j: j.view(), lambda j, r: "id", id="view-name"),
        pytest.param(lambda j: j, lambda j, r: "id", id="self-name"),
        pytest.param(lambda j: j, lambda j, r: [("id", "id")], id="self-pair"),
    ],
)
def test_join_chain_to_itself(make_right, make_predicate):
    a = xo.memtable({"id": [1, 2], "x": ["p", "q"]})
    b = xo.memtable({"id": [1, 2], "y": [10, 20]})
    j = a.join(b, "id")
    r = make_right(j)
    expr = j.left_join(r, make_predicate(j, r))
    df = expr.execute().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df[["id", "id_right"]].astype("float64"),
        pd.DataFrame({"id": [1, 2], "id_right": [1, 2]}, dtype="float64"),
    )


def test_join_on_field_of_nested_join_chain():
    # `j.flag` is a literal inside `j`, which sits on the outer side of a left
    # join; it must bind to that join's output, which is null for `c.id == 3`
    a = xo.memtable({"id": [1, 2]})
    b = xo.memtable({"id": [1, 2]})
    c = xo.memtable({"id": [1, 3]})
    m = xo.memtable({"flag": [1]})
    j = a.join(b, "id").select("id", flag=xo.literal(1))
    expr = c.left_join(j, c.id == j.id).left_join(m, j.flag == m.flag)
    df = expr.execute().sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df[["id", "flag_right"]].astype("float64"),
        pd.DataFrame({"id": [1, 3], "flag_right": [1, None]}, dtype="float64"),
    )


@pytest.mark.parametrize(
    "extend,expected",
    [
        pytest.param(
            lambda q, j: q.left_join(xo.memtable({"pid": [-1, 2]}), [("pub", "pid")]),
            {"pub": [-1, 1, 2], "pid": [-1, None, 2]},
            id="pair",
        ),
        pytest.param(
            lambda q, j: q.left_join(
                s := j.filter(j.pub == 2), q.pub == s.pub, rname="{name}_second"
            ),
            {"pub": [-1, 1, 2], "pub_second": [None, None, 2]},
            id="second-copy",
        ),
    ],
)
def test_extend_join_chain_to_its_own_filtered_result(extend, expected):
    # the chain now holds a table derived from itself; the computed key keeps
    # binding to the chain's own `pub` when the chain is extended
    j = pub_over_join("outer")
    q = j.left_join(j.filter(j.pub == 1), [("pub", "pub")])
    df = extend(q, j).execute().sort_values("pub").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df[list(expected)].astype("float64"), pd.DataFrame(expected, dtype="float64")
    )


@pytest.mark.parametrize(
    "make_predicate",
    [
        pytest.param(lambda j, r: j.pub == r.pub, id="value"),
        pytest.param(lambda j, r: "pub", id="name"),
    ],
)
def test_select_from_join_chain_to_its_own_filtered_result(make_predicate):
    # selecting from the join dereferences the chain's computed `pub` again;
    # it must keep binding to the chain, not to the table derived from it
    j = pub_over_join("outer")
    r = j.filter(j.pub == 1)
    expr = j.left_join(r, make_predicate(j, r)).select("pub", "pub_right")
    df = expr.execute().sort_values("pub").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df.astype("float64"),
        pd.DataFrame(
            {"pub": [-1, 1, 2], "pub_right": [None, 1, None]}, dtype="float64"
        ),
    )
