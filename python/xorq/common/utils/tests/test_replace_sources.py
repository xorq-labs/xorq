import pandas as pd
import pyarrow.compute as pc
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import ParquetCache, SourceCache
from xorq.common.utils.graph_utils import (
    _find_missing_tables,
    _namespace_to_database,
    find_all_sources,
    replace_sources,
)


def assert_result_equal(left, right):
    """Compare DataFrames ignoring row order (backends may return different orderings)."""
    left = left.sort_values(left.columns.tolist()).reset_index(drop=True)
    right = right.sort_values(right.columns.tolist()).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


@xo.udf.scalar.pyarrow
def double_int(arr: dt.int64) -> dt.int64:
    return pc.multiply(arr, 2)


@pytest.fixture(scope="session")
def parquet_path(parquet_dir):
    return parquet_dir / "batting.parquet"


BACKEND_NAMES = ("xorq_datafusion", "duckdb", "pandas")


def make_con(name):
    return getattr(xo, name).connect()


def backend_pair_id(val):
    return f"{val[0]}->{val[1]}"


BACKEND_PAIRS = tuple(
    (from_name, to_name) for from_name in BACKEND_NAMES for to_name in BACKEND_NAMES
)


@pytest.fixture(params=BACKEND_PAIRS, ids=[backend_pair_id(p) for p in BACKEND_PAIRS])
def from_to(request):
    return request.param


# ---------------------------------------------------------------------------
# Parametrized: registered table (requires transfer_tables=True)
# ---------------------------------------------------------------------------


def test_replace_registered_table(parquet_path, from_to):
    """Replace source on a table registered via read_parquet."""
    from_name, to_name = from_to
    from_con = make_con(from_name)
    to_con = make_con(to_name)

    t = from_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(from_con): to_con}, t, transfer_tables=True)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is to_con
    assert_result_equal(result.execute(), t.execute())


def test_replace_registered_table_raises_without_transfer(parquet_path):
    """Replacing a registered table without transfer_tables raises."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")

    ddb_con = xo.duckdb.connect()
    with pytest.raises(ValueError, match="DatabaseTable"):
        replace_sources({id(xo_con): ddb_con}, t)


# ---------------------------------------------------------------------------
# Parametrized: deferred read (no transfer needed)
# ---------------------------------------------------------------------------


def test_replace_deferred_read(parquet_path, from_to):
    """Replace source on a deferred_read_parquet expression."""
    from_name, to_name = from_to
    from_con = make_con(from_name)
    to_con = make_con(to_name)

    t = xo.deferred_read_parquet(parquet_path, from_con, table_name="dr_batting")
    result = replace_sources({id(from_con): to_con}, t)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is to_con
    assert isinstance(result.op(), rel.Read)
    assert result.op().source is to_con
    assert_result_equal(result.execute(), t.execute())


# ---------------------------------------------------------------------------
# Parametrized: projection and filter (requires transfer_tables=True)
# ---------------------------------------------------------------------------


def test_replace_with_projection(parquet_path, from_to):
    """Replace source on an expression with column selections and filters."""
    from_name, to_name = from_to
    from_con = make_con(from_name)
    to_con = make_con(to_name)

    t = from_con.read_parquet(parquet_path, table_name="batting")
    expr = t.filter(t.yearID > 2000).select("playerID", "yearID", "G")

    result = replace_sources({id(from_con): to_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is to_con
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Parametrized: into_backend
# ---------------------------------------------------------------------------


def test_replace_into_backend_target(parquet_path, from_to):
    """Replace the target backend of an into_backend expression."""
    from_name, to_name = from_to
    if from_name == "pandas" or to_name == "pandas":
        pytest.skip("pandas does not support into_backend")

    source_con = make_con(from_name)
    target_con = make_con(to_name)

    t = source_con.read_parquet(parquet_path, table_name="batting")
    expr = t.into_backend(target_con)

    replacement_con = make_con(to_name)
    result = replace_sources({id(target_con): replacement_con}, expr)

    sources = find_all_sources(result)
    assert replacement_con in sources
    assert target_con not in sources
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Parametrized: aggregation (requires transfer_tables=True)
# ---------------------------------------------------------------------------


def test_replace_with_aggregation(parquet_path, from_to):
    """Replace source on an expression with aggregation."""
    from_name, to_name = from_to
    from_con = make_con(from_name)
    to_con = make_con(to_name)

    t = from_con.read_parquet(parquet_path, table_name="batting")
    expr = t.group_by("yearID").agg(total_G=t.G.sum())

    result = replace_sources({id(from_con): to_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is to_con
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Parametrized: self-join (requires transfer_tables=True)
# ---------------------------------------------------------------------------


def test_replace_join_same_backend(parquet_path, from_to):
    """Replace source in a self-join on the same backend."""
    from_name, to_name = from_to
    from_con = make_con(from_name)
    to_con = make_con(to_name)

    t = from_con.read_parquet(parquet_path, table_name="batting")
    expr = t.join(t, predicates=["playerID", "yearID"])

    result = replace_sources({id(from_con): to_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert all(s is to_con for s in sources)
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Parametrized: scalar UDF (requires transfer_tables=True)
# ---------------------------------------------------------------------------


def test_replace_with_scalar_udf(parquet_path, from_to):
    """Replace source on an expression that applies a scalar UDF."""
    from_name, to_name = from_to
    if "pandas" in (from_name, to_name):
        pytest.skip("pandas cannot execute pyarrow scalar UDFs")
    from_con = make_con(from_name)
    to_con = make_con(to_name)

    t = from_con.read_parquet(parquet_path, table_name="batting")
    expr = t.mutate(doubled_G=double_int(t.G))

    result = replace_sources({id(from_con): to_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is to_con
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Caching (xorq-specific, not parametrized; requires transfer_tables=True)
# ---------------------------------------------------------------------------


def test_replace_source_with_source_cache(parquet_path):
    """Replace source when expression uses SourceCache."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.cache(SourceCache.from_kwargs(source=xo_con))

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert all(s is ddb_con for s in sources)
    cache_op = result.op()
    assert isinstance(cache_op, rel.CachedNode)
    assert cache_op.source is ddb_con
    assert_result_equal(result.execute(), expr.execute())


def test_replace_source_with_parquet_cache(parquet_path):
    """Replace source when expression uses ParquetCache."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.filter(t.yearID > 2000).cache(ParquetCache.from_kwargs(source=xo_con))

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert ddb_con in sources
    assert xo_con not in sources
    assert_result_equal(result.execute(), expr.execute())


def test_replace_source_with_chained_cache(parquet_path):
    """Replace source through a chain of cached expressions."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    step1 = t.filter(t.yearID > 2000).cache()
    expr = step1.filter(step1.G > 10).cache()

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert xo_con not in sources
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Multi-backend (not parametrized — fixed backend combinations)
# ---------------------------------------------------------------------------


def test_replace_one_of_two_backends(parquet_path):
    """Replace only one backend in a multi-source expression."""
    xo_con = xo.connect()
    ddb_con = xo.duckdb.connect()

    t1 = xo_con.read_parquet(parquet_path, table_name="batting")
    ddb_con.read_parquet(parquet_path, table_name="batting")
    t2 = ddb_con.table("batting")

    expr = t1.into_backend(ddb_con).join(t2, predicates=["playerID", "yearID"])

    ddb_con2 = xo.duckdb.connect()
    result = replace_sources({id(ddb_con): ddb_con2}, expr, transfer_tables=True)

    sources = find_all_sources(result)
    assert ddb_con not in sources
    assert xo_con in sources
    assert ddb_con2 in sources
    assert_result_equal(result.execute(), expr.execute())


def test_replace_both_backends(parquet_path):
    """Replace both backends in a multi-source expression."""
    xo_con = xo.connect()
    ddb_con = xo.duckdb.connect()

    t1 = xo_con.read_parquet(parquet_path, table_name="batting")
    ddb_con.read_parquet(parquet_path, table_name="batting")
    t2 = ddb_con.table("batting")

    expr = t1.into_backend(ddb_con).join(t2, predicates=["playerID", "yearID"])

    xo_con2 = xo.connect()
    ddb_con2 = xo.duckdb.connect()
    result = replace_sources(
        {id(xo_con): xo_con2, id(ddb_con): ddb_con2}, expr, transfer_tables=True
    )

    sources = find_all_sources(result)
    assert xo_con not in sources
    assert ddb_con not in sources
    assert xo_con2 in sources
    assert ddb_con2 in sources
    assert_result_equal(result.execute(), expr.execute())


def test_replace_source_in_into_backend_chain(parquet_path):
    """Replace source through xorq -> duckdb -> xorq chain."""
    xo_con1 = xo.connect()
    ddb_con = xo.duckdb.connect()
    xo_con2 = xo.connect()

    t = xo_con1.read_parquet(parquet_path, table_name="batting")
    step1 = t.into_backend(ddb_con)
    expr = step1.into_backend(xo_con2)

    xo_con3 = xo.connect()
    result = replace_sources({id(xo_con2): xo_con3}, expr)

    sources = find_all_sources(result)
    assert xo_con3 in sources
    assert xo_con2 not in sources
    assert xo_con1 in sources
    assert_result_equal(result.execute(), expr.execute())


# ---------------------------------------------------------------------------
# Edge cases (not parametrized)
# ---------------------------------------------------------------------------


def test_replace_sources_empty_mapping(parquet_path):
    """Empty mapping returns an equivalent expression."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.filter(t.yearID > 2000)

    result = replace_sources({}, expr)
    assert result.equals(expr)
    assert_result_equal(result.execute(), expr.execute())


def test_replace_sources_no_matching_source(parquet_path):
    """Mapping that doesn't match any source returns equivalent expression."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(ddb_con): xo.connect()}, t)
    assert result.equals(t)
    assert_result_equal(result.execute(), t.execute())


def test_replace_sources_preserves_schema(parquet_path):
    """Replacing source preserves the expression schema."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    original_schema = t.schema()

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con}, t, transfer_tables=True)

    assert result.schema() == original_schema
    assert_result_equal(result.execute(), t.execute())


def test_replace_sources_preserves_columns(parquet_path):
    """Replacing source preserves column names."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    original_columns = t.columns

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con}, t, transfer_tables=True)

    assert result.columns == original_columns
    assert_result_equal(result.execute(), t.execute())


def test_replace_sources_does_not_mutate_original(parquet_path):
    """Replacing source does not mutate the original expression."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    original_sources = find_all_sources(t)

    ddb_con = xo.duckdb.connect()
    replace_sources({id(xo_con): ddb_con}, t, transfer_tables=True)

    assert find_all_sources(t) == original_sources
    assert original_sources[0] is xo_con


# ---------------------------------------------------------------------------
# _namespace_to_database helper
# ---------------------------------------------------------------------------


def test_namespace_to_database_both():
    ns = ops.Namespace(catalog="my_cat", database="my_db")
    assert _namespace_to_database(ns) == ("my_cat", "my_db")


def test_namespace_to_database_db_only():
    ns = ops.Namespace(catalog=None, database="my_db")
    assert _namespace_to_database(ns) == "my_db"


def test_namespace_to_database_empty():
    ns = ops.Namespace()
    assert _namespace_to_database(ns) is None


# ---------------------------------------------------------------------------
# _find_missing_tables with namespace
# ---------------------------------------------------------------------------


def test_find_missing_tables_respects_namespace():
    """A table in a non-default schema should not be flagged as missing."""
    con = xo.duckdb.connect()
    con.raw_sql("CREATE SCHEMA IF NOT EXISTS other_schema")
    con.raw_sql("CREATE TABLE other_schema.my_table AS SELECT 1 AS x")

    ns = ops.Namespace(catalog=None, database="other_schema")
    result = _find_missing_tables([(con, con, "my_table", ns)])
    assert result == []


def test_find_missing_tables_detects_truly_missing():
    """A table that doesn't exist anywhere should be flagged as missing."""
    con = xo.duckdb.connect()
    ns = ops.Namespace(catalog=None, database="nonexistent_schema")
    result = _find_missing_tables([(con, con, "no_such_table", ns)])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Cross-schema / cross-catalog replace_sources (end-to-end)
# ---------------------------------------------------------------------------


def test_replace_sources_cross_schema_no_transfer():
    """Replacing source on a table in a non-default schema should not
    require transfer_tables when the table already exists on the target."""
    con1 = xo.duckdb.connect()
    con1.raw_sql("CREATE SCHEMA IF NOT EXISTS alt")
    con1.raw_sql("CREATE TABLE alt.t AS SELECT 1 AS x")

    t = con1.table("t", database="alt")

    con2 = xo.duckdb.connect()
    con2.raw_sql("CREATE SCHEMA IF NOT EXISTS alt")
    con2.raw_sql("CREATE TABLE alt.t AS SELECT 1 AS x")

    # Before fix: raises ValueError because list_tables() (default schema) misses "t"
    result = replace_sources({id(con1): con2}, t)
    assert_result_equal(result.execute(), t.execute())


def test_replace_sources_catalog_and_schema():
    """Table in catalog.schema is found via namespace, not bare list_tables."""
    con = xo.duckdb.connect()
    con.raw_sql("ATTACH ':memory:' AS other_cat")
    con.raw_sql("CREATE SCHEMA other_cat.my_schema")
    con.raw_sql("CREATE TABLE other_cat.my_schema.t AS SELECT 42 AS val")

    t = con.table("t", database=("other_cat", "my_schema"))

    con2 = xo.duckdb.connect()
    con2.raw_sql("ATTACH ':memory:' AS other_cat")
    con2.raw_sql("CREATE SCHEMA other_cat.my_schema")
    con2.raw_sql("CREATE TABLE other_cat.my_schema.t AS SELECT 42 AS val")

    result = replace_sources({id(con): con2}, t)
    assert result.execute()["val"].tolist() == [42]
