from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.caching import ParquetCache, SourceCache


@pytest.fixture
def ddb_batting(data_dir):
    ddb = xo.duckdb.connect()
    path = str(data_dir.joinpath("parquet/batting.parquet"))
    return ddb.read_parquet(path)


def test_into_duckdb(batting):
    """Pull data from Databricks into DuckDB and execute."""
    ddb = xo.duckdb.connect()
    result = batting.limit(20).into_backend(ddb).execute()
    assert len(result) == 20
    assert "playerID" in result.columns


def test_into_duckdb_with_filter(diamonds):
    """Filter in Databricks, materialise in DuckDB."""
    ddb = xo.duckdb.connect()
    result = (
        diamonds.filter(diamonds.cut == "Ideal").limit(10).into_backend(ddb).execute()
    )
    assert len(result) == 10
    assert (result["cut"] == "Ideal").all()


def test_into_databricks_from_duckdb(con, ddb_batting):
    """Push a small DuckDB table into Databricks and execute."""
    t = ddb_batting
    result = t.filter(t.HR > 40).limit(10).into_backend(con).execute()
    assert len(result) == 10
    assert (result["HR"] > 40).all()


def test_parquet_cache(con, ddb_batting, tmp_path):
    """Cache a DuckDB expr as parquet and re-execute from cache."""
    t = ddb_batting
    cache = ParquetCache.from_kwargs(source=con, relative_path=tmp_path)
    expr = t.limit(30)

    cached_expr = expr.cache(cache)
    assert len(cached_expr.execute()) == 30
    assert cache.exists(expr)


def test_source_cache_in_databricks(con, ddb_batting):
    """Push DuckDB data into Databricks, cache result back in Databricks."""
    t = ddb_batting
    expr = (
        t.filter(t.HR > 30)
        .select("playerID", "yearID", "HR")
        .limit(20)
        .into_backend(con)
        .cache(SourceCache.from_kwargs(source=con))
    )
    result = expr.execute()
    assert len(result) > 0
    assert (result["HR"] > 30).all()


def test_into_duckdb_then_parquet_cache(con, ddb_batting, tmp_path):
    t = ddb_batting
    ddb2 = xo.duckdb.connect()
    expr = (
        t.limit(25)
        .into_backend(ddb2)
        .cache(ParquetCache.from_kwargs(source=con, relative_path=tmp_path))
    )
    result = expr.execute()
    assert len(result) == 25
