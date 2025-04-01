import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

from xorq.expr.translate import plan_to_ibis
from xorq.internal import ContextProvider
from xorq.sql import parser
from xorq.vendor import ibis


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


@pytest.mark.parametrize(
    "sql",
    [
        "select t.a, t.b from t where t.c > 2",
        "select t.a, t.b from t where t.c > 2 limit 3",
        "select t.a, t.b * 2.5 from t",
        "select t.a, t.b / 2.5 from t",
    ],
)
def test_table_scan(con, sql, t):
    # from sql to LogicalPlan
    plan = parser.parse_sql(sql, ContextProvider({"t": t.schema().to_pyarrow()}))

    # from LogicalPlan back to ibis
    expr = plan_to_ibis(plan, {"t": t.schema()})

    expected = con.table("t").sql(sql).execute()
    actual = con.execute(expr)
    actual.columns = expected.columns

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "sql",
    [
        "select s.a from s where s.g is true",
        "select s.f from s where s.g is false",
        "select s.f from s where s.g is null",
        "select s.f from s where s.g is not null",
    ],
)
def test_table_scan_bool(con, sql, s):
    # from sql to LogicalPlan
    plan = parser.parse_sql(sql, ContextProvider({"s": s.schema().to_pyarrow()}))

    # from LogicalPlan back to ibis
    expr = plan_to_ibis(plan, {"s": s.schema()})

    expected = con.table("s").sql(sql).execute()
    actual = con.execute(expr)
    actual.columns = expected.columns

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.xfail(reason="datafusion 42.0.0 update introduced a bug")
@pytest.mark.parametrize(
    "sql",
    [
        "select t.a, t.b from t order by t.d",
        "select t.a, t.b from t order by t.d limit 5",
    ],
)
def test_sort(con, sql, t):
    # from sql to LogicalPlan
    plan = parser.parse_sql(sql, ContextProvider({"t": t.schema().to_pyarrow()}))

    # from LogicalPlan back to ibis
    expr = plan_to_ibis(plan, {"t": t.schema()})

    expected = con.table("t").sql(sql).execute()
    actual = con.execute(expr)
    actual.columns = expected.columns

    assert expr is not None
    assert_frame_equal(expected, actual)


def test_agg(con, t):
    sql = "select sum(t.b) from t"

    # from sql to LogicalPlan
    plan = parser.parse_sql(sql, ContextProvider({"t": t.schema().to_pyarrow()}))

    # from LogicalPlan back to ibis
    expr = plan_to_ibis(plan, {"t": t.schema()})

    expected = con.table("t").sql(sql).execute()
    actual = con.execute(expr)
    actual.columns = expected.columns

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "sql",
    [
        "select s.a from (select t.a from t) as s",
    ],
)
def test_subquery_alias(con, sql, t):
    # from sql to LogicalPlan
    plan = parser.parse_sql(sql, ContextProvider({"t": t.schema().to_pyarrow()}))

    # from LogicalPlan back to ibis
    expr = plan_to_ibis(plan, {"t": t.schema()})

    expected = con.table("t").sql(sql).execute()
    actual = con.execute(expr)
    actual.columns = expected.columns

    assert expr is not None
    assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "sql",
    [
        "select t.a, s.f + t.b from t join s on t.a = s.a",
        "select t.a, s.f - t.b from t left join s on t.a = s.a",
        "select t.a, s.f - t.b from t right join s on t.a = s.a",
        "select t.a, t.b, s.f from t join s on t.a = s.a where t.b > 1.5 limit 2",
    ],
)
def test_join(con, sql, t, s):
    # from sql to LogicalPlan
    plan = parser.parse_sql(
        sql,
        ContextProvider({"t": t.schema().to_pyarrow(), "s": s.schema().to_pyarrow()}),
    )

    # from LogicalPlan back to ibis
    expr = plan_to_ibis(plan, {"t": t.schema(), "s": s.schema()})

    expected = con.table("t").sql(sql).execute()
    actual = con.execute(expr)
    actual.columns = expected.columns

    assert expr is not None
    assert_frame_equal(expected, actual)
