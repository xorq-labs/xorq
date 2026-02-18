"""GizmoSQL backend tests — single-backend and multi-engine."""

from __future__ import annotations

from operator import methodcaller

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache, SourceCache


pytestmark = pytest.mark.gizmosql


# ── Single-backend tests ────────────────────────────────────────────────────


def test_connect(con):
    """Verify that the GizmoSQL connection is established."""
    assert con is not None
    assert con.name == "gizmosql"


def test_version(con):
    """Verify we can retrieve the server version."""
    version = con.version
    assert isinstance(version, str)
    assert len(version) > 0


def test_list_tables(con):
    """Verify that loaded test tables are listed."""
    tables = con.list_tables()
    assert isinstance(tables, list)
    assert "functional_alltypes" in tables
    assert "batting" in tables


def test_list_catalogs(con):
    """Verify that catalogs can be listed."""
    catalogs = con.list_catalogs()
    assert isinstance(catalogs, list)
    assert len(catalogs) > 0


def test_list_databases(con):
    """Verify that databases (schemas) can be listed."""
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert "main" in databases


def test_current_catalog(con):
    """Verify current_catalog property works."""
    catalog = con.current_catalog
    assert isinstance(catalog, str)
    assert len(catalog) > 0


def test_current_database(con):
    """Verify current_database property works."""
    database = con.current_database
    assert isinstance(database, str)
    assert database == "main"


def test_execute(con, alltypes):
    """Verify executing an expression returns a pandas DataFrame."""
    result = alltypes.limit(10).execute()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10


def test_to_pyarrow(con, alltypes):
    """Verify to_pyarrow returns a PyArrow Table."""
    result = con.to_pyarrow(alltypes.limit(10))
    assert isinstance(result, pa.Table)
    assert result.num_rows == 10


def test_to_pyarrow_batches(con, alltypes):
    """Verify to_pyarrow_batches returns a RecordBatchReader."""
    reader = con.to_pyarrow_batches(alltypes.limit(10))
    assert isinstance(reader, pa.ipc.RecordBatchReader)
    table = reader.read_all()
    assert table.num_rows == 10


def test_create_and_drop_table(con, temp_table):
    """Verify table creation and deletion via DDL."""
    t = xo.memtable({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    con.create_table(temp_table, t)

    assert temp_table in con.list_tables()

    result = con.table(temp_table).execute()
    assert len(result) == 3

    with con._safe_raw_sql(f'DROP TABLE IF EXISTS "{temp_table}"'):
        pass
    assert temp_table not in con.list_tables()


def test_read_parquet(con):
    """Verify read_parquet via _read_local_and_ingest."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    parquet_path = root / "ci" / "ibis-testing-data" / "parquet" / "diamonds.parquet"
    if not parquet_path.exists():
        pytest.skip("Test data not available")

    t = con.read_parquet(parquet_path, table_name="test_diamonds_read")
    result = t.limit(5).execute()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # Clean up
    with con._safe_raw_sql('DROP TABLE IF EXISTS "test_diamonds_read"'):
        pass


def test_memtable(con):
    """Verify in-memory table registration via ADBC."""
    t = xo.memtable({"x": [10, 20, 30], "y": ["a", "b", "c"]})
    result = con.execute(t)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result["x"]) == [10, 20, 30]


def test_schema(con, alltypes):
    """Verify we can retrieve the schema of a table."""
    schema = alltypes.schema()
    assert len(schema) > 0
    assert "id" in schema


# ── into_backend tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["execute", "to_pyarrow", "to_pyarrow_batches"])
def test_into_backend_to_duckdb(con, batting, method):
    """GizmoSQL → DuckDB via into_backend, all output methods."""
    ddb_con = xo.duckdb.connect()
    expr = batting.limit(50).into_backend(ddb_con, "gz_batting")
    res = methodcaller(method)(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = res.read_all()
    if isinstance(res, pa.Table):
        assert res.num_rows == 50
    else:
        assert len(res) == 50


@pytest.mark.parametrize("method", ["execute", "to_pyarrow", "to_pyarrow_batches"])
def test_into_backend_to_datafusion(con, batting, method):
    """GizmoSQL → DataFusion (xorq default) via into_backend."""
    xorq_con = xo.connect()
    expr = batting.limit(50).into_backend(xorq_con, "gz_batting")
    res = methodcaller(method)(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = res.read_all()
    if isinstance(res, pa.Table):
        assert res.num_rows == 50
    else:
        assert len(res) == 50


def test_into_backend_join_on_duckdb(con, batting, awards_players):
    """Transfer two GizmoSQL tables to DuckDB, join there."""
    ddb_con = xo.duckdb.connect()

    batting_ddb = batting.filter(batting.yearID >= 2010).into_backend(
        ddb_con, "gz_batting"
    )
    awards_ddb = awards_players.into_backend(ddb_con, "gz_awards")

    expr = (
        batting_ddb.join(awards_ddb, predicates=["playerID", "yearID"])
        .limit(20)
        .select("playerID", "yearID", "awardID", "H")
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 20


def test_into_backend_double_hop(con, batting):
    """GizmoSQL → DuckDB → DataFusion (two hops)."""
    from xorq.vendor.ibis import _

    ddb_con = xo.duckdb.connect()
    xorq_con = xo.connect()

    expr = (
        batting.filter(batting.yearID == 2015)
        .select("playerID", "yearID", "teamID", "H")
        .into_backend(ddb_con, "gz_hop1")
        .filter(_.H > 100)
        .into_backend(xorq_con, "gz_hop2")
        .limit(10)
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 10


def test_into_backend_reverse_duckdb_to_gizmosql(con, batting):
    """DuckDB → GizmoSQL: push local data into the GizmoSQL server."""
    ddb_con = xo.duckdb.connect()
    local_table = ddb_con.create_table(
        "local_players",
        batting.limit(100).to_pyarrow(),
    )
    expr = local_table.into_backend(con, "ddb_to_gz")

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 100


# ── Caching tests ───────────────────────────────────────────────────────────


def test_into_backend_source_cache(con, batting):
    """GizmoSQL → DataFusion with SourceCache."""
    xorq_con = xo.connect()

    expr = (
        batting.filter(batting.yearID >= 2014)
        .limit(50)
        .into_backend(xorq_con, "gz_cached_batting")
        .cache(SourceCache.from_kwargs(source=xorq_con))
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 50


def test_into_backend_parquet_cache(con, batting, tmp_path):
    """GizmoSQL → DuckDB with ParquetCache."""
    ddb_con = xo.duckdb.connect()

    expr = (
        batting.filter(batting.yearID >= 2014)
        .limit(50)
        .into_backend(ddb_con, "gz_pq_batting")
        .cache(ParquetCache.from_kwargs(source=ddb_con, relative_path=tmp_path))
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 50

    # second execution should hit cache
    res2 = expr.execute()
    assert len(res2) == len(res)


def test_into_backend_chained_caches(con, batting, tmp_path):
    """GizmoSQL → DataFusion [SourceCache] → DuckDB [ParquetCache]."""
    xorq_con = xo.connect()
    ddb_con = xo.duckdb.connect()

    expr = (
        batting.filter(batting.yearID >= 2014)
        .limit(50)
        .into_backend(xorq_con, "gz_chain_batting")
        .cache(SourceCache.from_kwargs(source=xorq_con))
        .into_backend(ddb_con)
        .select("playerID", "yearID", "teamID", "H")
        .cache(ParquetCache.from_kwargs(source=ddb_con, relative_path=tmp_path))
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 50
