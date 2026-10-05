import pyarrow as pa
import pytest

import xorq.api as xo


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
