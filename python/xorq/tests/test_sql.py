import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops


@pytest.mark.parametrize("file_format", ["parquet", "csv"])
def test_sql_on_deferred_read(file_format, request):
    ddb = xo.duckdb.connect()
    diamonds_path = (
        request.getfixturevalue(f"{file_format}_dir") / f"diamonds.{file_format}"
    )

    deferred_read = getattr(xo, f"deferred_read_{file_format}")

    expr = (
        deferred_read(diamonds_path, con=ddb, table_name="diamonds_ddb")
        .limit(2)
        .select(xo._.y, xo._.z, xo._.cut)
        .sql("SELECT cut, avg(y + z) as c FROM diamonds_ddb group by cut")
    )

    assert not expr.execute().empty


@pytest.fixture(
    params=[
        "xorq_datafusion",
        "datafusion",
        "sqlite",
        pytest.param("postgres", marks=pytest.mark.postgres),
    ]
)
def int_ratio_expr(request):
    if request.param == "postgres":
        con = request.getfixturevalue("pg")
    else:
        con = getattr(xo, request.param).connect()
    t = con.create_table(
        "int_ratio",
        pa.table(
            {"a": pa.array([1500], pa.int64()), "b": pa.array([340000000], pa.int64())}
        ),
    )
    return t.mutate(ratio=t.a / t.b, ratio_cast=t.a.cast("float64") / t.b)


def test_integer_division_is_true_division(int_ratio_expr):
    (row,) = int_ratio_expr.execute().to_dict("records")
    assert row["ratio"] == pytest.approx(1500 / 340000000)
    assert row["ratio"] == row["ratio_cast"]


def test_compile_renders_the_same_sql_as_a_fresh_tree(int_ratio_expr):
    """TYPED_DIVISION dialects mutate the shared tree while rendering (#2367)."""
    con = int_ratio_expr._find_backend()
    compiler = con.compiler
    fresh = compiler.to_sqlglot(int_ratio_expr).sql(dialect=compiler.dialect)
    assert con.compile(int_ratio_expr) == fresh


@pytest.fixture(params=["sqlite", "duckdb", "xorq_datafusion"])
def column_subset_table(request):
    """A DatabaseTable bound to two of its three physical columns (#2421)."""
    con = getattr(xo, request.param).connect()
    con.create_table("t", pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": ["x", "y"]}))
    return ops.DatabaseTable(
        name="t", schema=xo.schema({"a": "int64", "b": "int64"}), source=con
    ).to_expr()


def test_select_every_bound_column_of_a_subset_table(column_subset_table):
    t = column_subset_table
    expr = t.select("a", "b")
    # the bound compile names the columns; `xo.to_sql` unbinds first and keeps `*`
    assert "*" not in expr._find_backend().compile(expr)
    assert expr.execute().columns.tolist() == ["a", "b"]
    assert t.filter(t.a > 1).execute().columns.tolist() == ["a", "b"]
    # a strict subset already compiled to an explicit column list
    assert t.select("a").execute().columns.tolist() == ["a"]


@pytest.fixture(params=["duckdb", "xorq_datafusion"])
def hopped_table(request):
    """A memtable hopped into a fresh backend under the name a query will use."""
    con = getattr(xo, request.param).connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    return xo.memtable(df, name="src").into_backend(con, "hopped")


@pytest.mark.parametrize(
    "to_named",
    [
        pytest.param(lambda t: (t, "hopped"), id="into_backend"),
        pytest.param(lambda t: (t.alias("h"), "h"), id="alias"),
        pytest.param(
            lambda t: (t.sql("SELECT a, b FROM hopped").alias("h"), "h"),
            id="chained",
        ),
    ],
)
def test_sql_after_into_backend(hopped_table, to_named):
    (t, name) = to_named(hopped_table)
    expr = t.sql(f"SELECT b, a + 1 AS c FROM {name} WHERE a > 1 ORDER BY a")
    assert expr.execute().to_dict("list") == {"b": ["y", "z"], "c": [3, 4]}


def test_sql_after_unnamed_into_backend(hopped_table):
    # an unnamed hop gets a generated placeholder name; alias it to query it
    t = xo.memtable(hopped_table.execute()).into_backend(hopped_table._find_backend())
    expr = t.alias("h").sql("SELECT b, a + 1 AS c FROM h WHERE a > 1 ORDER BY a")
    assert expr.execute().to_dict("list") == {"b": ["y", "z"], "c": [3, 4]}


@pytest.mark.parametrize("backend", ["duckdb", "xorq_datafusion"])
@pytest.mark.parametrize(
    "to_child",
    [
        pytest.param(lambda t: t, id="plain"),
        # a shared subexpression compiles to a CTE of the child, which the
        # query hoists to its own WITH and strips from the child
        pytest.param(
            lambda t: (
                (f := t.filter(t.a > 0))
                .join(f.view(), "a", rname="{name}_r")
                .select("a", "b")
            ),
            id="child_with_cte",
        ),
        pytest.param(
            lambda t: (
                t.sql("SELECT a, b FROM src").alias("u").sql("SELECT a, b FROM u")
            ),
            id="nested_sql",
        ),
    ],
)
def test_sql_hoists_child_ctes(backend, to_child):
    con = getattr(xo, backend).connect()
    t = con.create_table(
        "src", pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), overwrite=True
    )
    expr = (
        to_child(t).alias("s").sql("SELECT b, a + 1 AS c FROM s WHERE a > 1 ORDER BY a")
    )
    assert expr.execute().to_dict("list") == {"b": ["y", "z"], "c": [3, 4]}
