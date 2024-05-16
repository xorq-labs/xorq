import ibis
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from letsql.optimizer import optimize_ibis, optimize_sql


@pytest.fixture(scope="session")
def con():
    return ibis.connect("duckdb://")


@pytest.fixture(scope="session")
def t(con):
    con.create_table(
        "t",
        pa.Table.from_pydict(
            {
                "a": ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "c": [1, 2, 3, 4, 5, 6, 7],
                "d": [5, 6, 1, 7, 2, 4, 3],
                "e": [1, 2, 3, 4, 5, 6, 7],
            }
        ),
    )

    return con.table("t")


@pytest.fixture(scope="session")
def s(con):
    con.create_table(
        "s",
        pa.Table.from_pydict(
            {
                "a": ["a1", "a2", "a3", "a4", "a5", "a9", "a8"],
                "f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "g": [True, False, False, False, False, True, True],
            }
        ),
    )

    return con.table("s")


@pytest.fixture(scope="session")
def g(con):
    con.create_table(
        "g",
        pa.Table.from_pydict(
            {
                "a": ["a1", "a2", "a1", "a2", "a1", "a1", "a2"],
                "f": [1.5, 2.0, -3.0, 4.7, -5.0, -6.0, 7.0],
                "g": [True, False, False, False, False, True, True],
            }
        ),
    )

    return con.table("g")


def test_simple_optimize(con, t):
    query = "select * from t where t.c > 3 + 2"
    expr = optimize_sql(query, {"t": t.schema()})

    expected = con.table("t").sql(query).execute()
    actual = con.execute(expr)

    assert expr is not None
    assert expr.op().parent.predicates[0].right.value == 5
    assert_frame_equal(expected, actual)


def test_logical_algebra_optimize(con, t):
    query = "select * from t where t.c > 3 and false"
    expr = optimize_sql(query, {"t": t.schema()})

    expected = con.table("t").sql(query).execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)


def test_roundtrip(con, t):
    original = t.select([t.a, t.b, t.c]).filter([t.c > 1])
    expr = optimize_ibis(original, {"t": t.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)


def test_roundtrip_agg(con, t):
    original = t.aggregate([t.b.sum()])
    expr = optimize_ibis(original, {"t": t.schema()}, dialect="duckdb")
    expected = original.execute()
    actual = con.execute(expr)

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "how",
    [
        "left",
        "right",
        "inner",
    ],
)
def test_roundtrip_join(con, t, s, how):
    original = t.join(s, "a", how=how)
    expr = optimize_ibis(original, {"t": t.schema(), "s": s.schema()}, dialect="duckdb")
    expected = original.execute()
    actual = con.execute(expr)

    assert expr is not None
    assert_frame_equal(expected, actual)


def test_roundtrip_group(con, g):
    original = g.group_by("a").aggregate(f_sum=g.f.sum(), f_mean=g.f.mean())
    expr = optimize_ibis(original, {"g": g.schema()}, dialect="duckdb")

    expected = original.execute().sort_values(by="a").reset_index(drop=True)
    actual = con.execute(expr).sort_values(by="a").reset_index(drop=True)

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "how",
    [
        "left",
        "right",
        "inner",
    ],
)
def test_roundtrip_join_with_filter(con, t, s, how):
    original = t.join(s, "a", how=how).filter(t.b > 3)
    expr = optimize_ibis(original, {"t": t.schema(), "s": s.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "condition",
    [
        lambda x: x.filter(x.b > 10),
        lambda x: x.filter(x.b >= 10),
        lambda x: x.filter(x.b > 1.5),
        lambda x: x.filter(x.b < 1_000_000),
        lambda x: x.filter(x.b <= 1_000_000),
        lambda x: x.filter(x.b == 2.0),
    ],
)
def test_roundtrip_filter(con, t, condition):
    original = condition(t.select([t.a, t.b, t.c]))
    expr = optimize_ibis(original, {"t": t.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "condition",
    [
        lambda x: x.filter(x.g),
        lambda x: x.filter(x.g.negate()),
        lambda x: x.filter(x.a.isin(["a1", "a2"])),
        lambda x: x.filter(x.a.isin(["a1", "a2"]) & x.g),
    ],
)
def test_roundtrip_boolean_filter(con, s, condition):
    original = condition(s.select([s.g, s.a]))
    expr = optimize_ibis(original, {"s": s.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: (x.b * 2).name("new_b"),
        lambda x: (x.b / 2).name("new_b"),
        lambda x: (x.b + 2).name("new_b"),
        lambda x: (x.b - 2).name("new_b"),
    ],
)
def test_roundtrip_arithmetic(con, t, operation):
    original = operation(t)
    expr = optimize_ibis(original, {"t": t.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr).squeeze()

    assert_series_equal(expected, actual)


@pytest.mark.xfail(reason="datafusion 37.1.0 introduced a bug in aggregate functions")
def test_roundtrip_nested_agg(con, g):
    original = (
        g.group_by(["a", "g"])
        .aggregate(the_sum=g.f.sum())
        .group_by("a")
        .aggregate(mad=lambda x: x.the_sum.abs().mean())
    )
    expr = optimize_ibis(original, {"g": g.schema()}, dialect="duckdb")

    # sorting is required to avoid shuffling
    expected = original.execute().sort_values(by="a").reset_index(drop=True)
    actual = con.execute(expr).sort_values(by="a").reset_index(drop=True)

    assert_frame_equal(expected, actual)


def test_roundtrip_all(con, t):
    original = t[t]
    expr = optimize_ibis(original, {"t": t.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(None, 3), (2, 3), (4, 0), (0, 3)],
)
def test_roundtrip_sort(con, g, limit, offset):
    original = g.order_by(g.f).limit(limit, offset=offset)
    expr = optimize_ibis(original, {"g": g.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)


def test_roundtrip_case(con, t):
    original = t.a.case().when("a1", 1).when("a2", 2).else_(3).end()
    expr = optimize_ibis(original, {"t": t.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr).squeeze()

    assert_series_equal(expected, actual)


def test_roundtrip_distinct(con, g):
    original = g.distinct()
    expr = optimize_ibis(original, {"g": g.schema()}, dialect="duckdb")

    # sort to bypass shuffling
    expected = original.execute().sort_values(by=["a", "f", "g"]).reset_index(drop=True)
    actual = con.execute(expr).sort_values(by=["a", "f", "g"]).reset_index(drop=True)

    assert_frame_equal(expected, actual)


def test_roundtrip_nunique(con, g):
    original = g.a.nunique()
    expr = optimize_ibis(original, {"g": g.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    # actual is a DataFrame and expected is a scalar value
    assert (actual == expected).to_numpy().all()


def test_roundtrip_topk(con, g):
    original = g.a.topk(3)
    expr = optimize_ibis(original, {"g": g.schema()}, dialect="duckdb")

    expected = original.execute()
    actual = con.execute(expr)

    assert_frame_equal(expected, actual)
