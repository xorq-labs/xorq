import pyarrow.compute as pc
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.relations as rel
from xorq.caching import ParquetCache, SourceCache
from xorq.common.utils.graph_utils import find_all_sources, replace_sources


@xo.udf.scalar.pyarrow
def double_int(arr: dt.int64) -> dt.int64:
    return pc.multiply(arr, 2)


@xo.udf.agg.builtin
def my_mean(arr: dt.float64) -> dt.float64:
    return pc.mean(arr)


@pytest.fixture(scope="session")
def parquet_path(parquet_dir):
    return parquet_dir / "batting.parquet"


# ---------------------------------------------------------------------------
# Replace xorq connection with duckdb
# ---------------------------------------------------------------------------


def test_replace_xorq_registered_table_with_duckdb(parquet_path):
    """Replace source on a table that was read_parquet into xorq."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")

    result = replace_sources({id(xo_con): ddb_con}, t)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is ddb_con


def test_replace_xorq_deferred_read_with_duckdb(parquet_path):
    """Replace source on a deferred_read_parquet expression."""
    xo_con = xo.connect()
    t = xo.deferred_read_parquet(parquet_path, xo_con, table_name="dr_batting")

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con}, t)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is ddb_con
    assert isinstance(result.op(), rel.Read)
    assert result.op().source is ddb_con


def test_replace_xorq_into_backend_with_duckdb(parquet_path):
    """Replace a source used as the target of into_backend."""
    xo_con = xo.connect()
    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    ddb_t = ddb_con.table("batting")

    expr = ddb_t.into_backend(xo_con)

    ddb_con2 = xo.duckdb.connect()
    result = replace_sources({id(xo_con): ddb_con2}, expr)

    sources = find_all_sources(result)
    assert ddb_con2 in sources
    assert xo_con not in sources


def test_replace_xorq_with_projection(parquet_path):
    """Replace source on an expression with column selections and filters."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.filter(t.yearID > 2000).select("playerID", "yearID", "G")

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")

    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is ddb_con


# ---------------------------------------------------------------------------
# Replace duckdb connection with xorq
# ---------------------------------------------------------------------------


def test_replace_duckdb_registered_table_with_xorq(parquet_path):
    """Replace duckdb source with xorq on a registered table."""
    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    t = ddb_con.table("batting")

    xo_con = xo.connect()
    result = replace_sources({id(ddb_con): xo_con}, t)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is xo_con


def test_replace_duckdb_deferred_read_with_xorq(parquet_path):
    """Replace duckdb source with xorq on a deferred read."""
    ddb_con = xo.duckdb.connect()
    t = xo.deferred_read_parquet(parquet_path, ddb_con, table_name="dr_batting")

    xo_con = xo.connect()
    result = replace_sources({id(ddb_con): xo_con}, t)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is xo_con


def test_replace_duckdb_into_backend_with_xorq(parquet_path):
    """Replace duckdb target in an into_backend chain."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")

    ddb_con = xo.duckdb.connect()
    expr = t.into_backend(ddb_con)

    xo_con2 = xo.connect()
    result = replace_sources({id(ddb_con): xo_con2}, expr)

    sources = find_all_sources(result)
    assert xo_con2 in sources
    assert ddb_con not in sources


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_replace_source_with_source_cache(parquet_path):
    """Replace source when expression uses SourceCache."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.cache(SourceCache.from_kwargs(source=xo_con))

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert all(s is ddb_con for s in sources)
    cache_op = result.op()
    assert isinstance(cache_op, rel.CachedNode)
    assert cache_op.source is ddb_con


def test_replace_source_with_parquet_cache(parquet_path):
    """Replace source when expression uses ParquetCache."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.filter(t.yearID > 2000).cache(ParquetCache.from_kwargs(source=xo_con))

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert ddb_con in sources
    assert xo_con not in sources


def test_replace_source_with_chained_cache(parquet_path):
    """Replace source through a chain of cached expressions."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    step1 = t.filter(t.yearID > 2000).cache()
    expr = step1.filter(step1.G > 10).cache()

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert xo_con not in sources


# ---------------------------------------------------------------------------
# UDF / UDAF
# ---------------------------------------------------------------------------


def test_replace_source_with_scalar_udf(parquet_path):
    """Replace source on an expression that applies a scalar UDF."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.mutate(doubled_G=double_int(t.G))

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is ddb_con


def test_replace_source_with_agg_udf(parquet_path):
    """Replace source on an expression that uses an aggregate UDF."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.group_by("yearID").agg(mean_G=my_mean(t.G.cast("float64")))

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is ddb_con


# ---------------------------------------------------------------------------
# Multi-backend / complex graphs
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
    result = replace_sources({id(ddb_con): ddb_con2}, expr)

    sources = find_all_sources(result)
    assert ddb_con not in sources
    assert xo_con in sources
    assert ddb_con2 in sources


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
    result = replace_sources({id(xo_con): xo_con2, id(ddb_con): ddb_con2}, expr)

    sources = find_all_sources(result)
    assert xo_con not in sources
    assert ddb_con not in sources
    assert xo_con2 in sources
    assert ddb_con2 in sources


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


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_replace_sources_empty_mapping(parquet_path):
    """Empty mapping returns an equivalent expression."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.filter(t.yearID > 2000)

    result = replace_sources({}, expr)
    assert result.equals(expr)


def test_replace_sources_no_matching_source(parquet_path):
    """Mapping that doesn't match any source returns equivalent expression."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")

    ddb_con = xo.duckdb.connect()
    result = replace_sources({id(ddb_con): xo.connect()}, t)
    assert result.equals(t)


def test_replace_sources_preserves_schema(parquet_path):
    """Replacing source preserves the expression schema."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    original_schema = t.schema()

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, t)

    assert result.schema() == original_schema


def test_replace_sources_preserves_columns(parquet_path):
    """Replacing source preserves column names."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    original_columns = t.columns

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, t)

    assert result.columns == original_columns


def test_replace_sources_does_not_mutate_original(parquet_path):
    """Replacing source does not mutate the original expression."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    original_sources = find_all_sources(t)

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    replace_sources({id(xo_con): ddb_con}, t)

    assert find_all_sources(t) == original_sources
    assert original_sources[0] is xo_con


def test_replace_sources_join_same_backend(parquet_path):
    """Replace source in a self-join on the same backend."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.join(t, predicates=["playerID", "yearID"])

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert all(s is ddb_con for s in sources)


def test_replace_sources_aggregation(parquet_path):
    """Replace source on an expression with aggregation."""
    xo_con = xo.connect()
    t = xo_con.read_parquet(parquet_path, table_name="batting")
    expr = t.group_by("yearID").agg(total_G=t.G.sum())

    ddb_con = xo.duckdb.connect()
    ddb_con.read_parquet(parquet_path, table_name="batting")
    result = replace_sources({id(xo_con): ddb_con}, expr)

    sources = find_all_sources(result)
    assert len(sources) == 1
    assert sources[0] is ddb_con
