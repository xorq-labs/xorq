import pytest
import sqlglot.expressions as sge

from xorq.vendor import ibis
from xorq.vendor.ibis.backends.sql.compilers.duckdb import DuckDBCompiler
from xorq.vendor.ibis.backends.sql.compilers.snowflake import SnowflakeCompiler


def _wide_table(n_cols=10):
    """Create an unbound table wide enough that DropColumns survives the rewrite.

    The ``drop_columns_to_select`` rewrite converts DropColumns to an explicit
    Select when >= 50% of columns are dropped. Using 10 columns and dropping 2
    keeps the DropColumns node intact so the compiler's ``visit_DropColumns``
    is exercised.
    """
    cols = {f"col_{i}": "int64" for i in range(n_cols)}
    return ibis.table(cols, name="wide_table")


@pytest.fixture(params=["duckdb", "snowflake"])
def dialect_and_compiler(request):
    compilers = {
        "duckdb": DuckDBCompiler,
        "snowflake": SnowflakeCompiler,
    }
    return request.param, compilers[request.param]()


def _compile(compiler, expr):
    queries = compiler.to_sqlglot(expr)
    return queries[0].sql(dialect=compiler.dialect)


def test_drop_columns_generates_exclude(dialect_and_compiler):
    """visit_DropColumns must produce SELECT * EXCLUDE (...), not bare SELECT *."""
    dialect, compiler = dialect_and_compiler
    t = _wide_table()
    expr = t.drop("col_0", "col_1")

    sql = _compile(compiler, expr)
    assert "EXCLUDE" in sql, (
        f"{dialect}: DropColumns generated bare SELECT * without EXCLUDE clause: {sql}"
    )


def test_drop_columns_lists_all_dropped(dialect_and_compiler):
    """Every dropped column must appear in the EXCLUDE clause."""
    _, compiler = dialect_and_compiler
    t = _wide_table()
    to_drop = ["col_0", "col_3"]
    expr = t.drop(*to_drop)

    sql = _compile(compiler, expr)
    for col in to_drop:
        assert col in sql, f"Dropped column {col!r} missing from SQL: {sql}"


def test_drop_columns_preserves_remaining(dialect_and_compiler):
    """Dropped columns must NOT appear in the result schema."""
    _, compiler = dialect_and_compiler
    t = _wide_table()
    to_drop = ["col_0", "col_1"]
    expr = t.drop(*to_drop)

    result_cols = set(expr.columns)
    for col in to_drop:
        assert col not in result_cols


def test_sge_star_except_underscore():
    """Regression: sge.Star must use 'except_' (with underscore), not 'except'.

    'except' is silently ignored by sqlglot because the arg slot is named
    'except_' (Python reserved word). This caused SELECT * EXCLUDE to render
    as bare SELECT *, making DropColumns a no-op.
    """
    import sqlglot as sg

    excludes = [sg.column("dropped_col", quoted=True)]

    star_broken = sge.Star(**{"except": excludes})
    star_fixed = sge.Star(**{"except_": excludes})

    sql_broken = sge.Column(this=star_broken).sql()
    sql_fixed = sge.Column(this=star_fixed).sql()

    assert "dropped_col" not in sql_broken, (
        "sqlglot now accepts 'except' — update the compiler if this changes"
    )
    assert "dropped_col" in sql_fixed
