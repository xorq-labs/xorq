"""Tests for GizmoSQL vendored backend coverage.

Covers methods and branches in the vendored backend that are not
exercised by test_basic.py (single-backend, into_backend, caching).
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from urllib.parse import ParseResult

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.common.exceptions as exc


pytestmark = pytest.mark.gizmosql


# ── _Settings ───────────────────────────────────────────────────────────────


def test_settings_getitem(con):
    """_Settings.__getitem__ returns a setting value."""
    tz = con.settings["TimeZone"]
    assert isinstance(tz, str)
    assert tz == "UTC"


def test_settings_getitem_missing(con):
    """_Settings.__getitem__ raises KeyError for unknown settings."""
    with pytest.raises(KeyError):
        con.settings["__nonexistent_setting__"]


def test_settings_repr(con):
    """_Settings.__repr__ returns a non-empty string."""
    r = repr(con.settings)
    assert isinstance(r, str)
    assert len(r) > 0


# ── create_table edge cases ─────────────────────────────────────────────────


def test_create_table_schema_only(con, temp_table):
    """create_table with schema but no obj creates an empty table."""
    con.create_table(temp_table, schema={"a": "int64", "b": "string"})
    assert temp_table in con.list_tables()
    result = con.table(temp_table).execute()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["a", "b"]


def test_create_table_overwrite(con, temp_table):
    """create_table with overwrite=True replaces existing table."""
    t1 = xo.memtable({"x": [1, 2]})
    con.create_table(temp_table, t1)
    assert con.table(temp_table).count().execute() == 2

    t2 = xo.memtable({"x": [10, 20, 30]})
    con.create_table(temp_table, t2, overwrite=True)
    assert con.table(temp_table).count().execute() == 3


def test_create_table_from_dict(con, temp_table):
    """create_table with a dict (non-Expr obj) is converted via memtable."""
    con.create_table(temp_table, {"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    result = con.table(temp_table).execute()
    assert len(result) == 3


def test_create_table_no_obj_no_schema_raises(con, temp_table):
    """create_table without obj or schema raises ValueError."""
    with pytest.raises(ValueError, match="Either `obj` or `schema`"):
        con.create_table(temp_table)


def test_create_table_null_type_raises(con, temp_table):
    """create_table with a NULL typed column raises XorqTypeError."""
    with pytest.raises(exc.XorqTypeError, match="NULL typed columns"):
        con.create_table(temp_table, schema={"x": "null"})


# ── create_database / drop_database ─────────────────────────────────────────


def test_create_and_drop_database(con):
    """create_database and drop_database manage schemas."""
    db_name = "test_temp_schema"
    try:
        con.create_database(db_name)
        assert db_name in con.list_databases()
    finally:
        with contextlib.suppress(Exception):
            con.drop_database(db_name)


def test_create_database_with_catalog_raises(con):
    """create_database with a catalog argument raises."""
    with pytest.raises(exc.UnsupportedOperationError):
        con.create_database("test_db", catalog="some_catalog")


def test_drop_database_with_wrong_catalog_raises(con):
    """drop_database with a different catalog raises."""
    with pytest.raises(exc.UnsupportedOperationError):
        con.drop_database("test_db", catalog="nonexistent_catalog")


# ── list_databases with catalog filter ──────────────────────────────────────


def test_list_databases_with_catalog(con):
    """list_databases with catalog= filters results."""
    dbs = con.list_databases(catalog="memory")
    assert isinstance(dbs, list)
    assert "main" in dbs


# ── get_schema error handling ───────────────────────────────────────────────


def test_get_schema_table_not_found(con):
    """get_schema raises TableNotFound for nonexistent tables."""
    with pytest.raises(exc.TableNotFound):
        con.get_schema("__this_table_does_not_exist__")


# ── _convert_kwargs ─────────────────────────────────────────────────────────


def test_convert_kwargs():
    """_convert_kwargs converts read_only string to bool."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    kwargs = {"read_only": "true"}
    Backend._convert_kwargs(kwargs)
    assert kwargs["read_only"] is True

    kwargs = {"read_only": "False"}
    Backend._convert_kwargs(kwargs)
    assert kwargs["read_only"] is False


# ── read_csv ────────────────────────────────────────────────────────────────


def test_read_csv(con):
    """read_csv reads a local CSV file via _read_local_and_ingest."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("id,name,value\n")
        f.write("1,alice,10.5\n")
        f.write("2,bob,20.3\n")
        f.write("3,charlie,30.1\n")
        csv_path = f.name

    try:
        t = con.read_csv(csv_path, table_name="test_read_csv")
        result = t.execute()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "name" in result.columns
    finally:
        with con._safe_raw_sql('DROP TABLE IF EXISTS "test_read_csv"'):
            pass
        Path(csv_path).unlink(missing_ok=True)


# ── read_json ───────────────────────────────────────────────────────────────


def test_read_json(con):
    """read_json reads newline-delimited JSON via _read_local_and_ingest."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        for row in [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "charlie"},
        ]:
            f.write(json.dumps(row) + "\n")
        json_path = f.name

    try:
        t = con.read_json(json_path, table_name="test_read_json")
        result = t.execute()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    finally:
        with con._safe_raw_sql('DROP TABLE IF EXISTS "test_read_json"'):
            pass
        Path(json_path).unlink(missing_ok=True)


# ── vendored execute (nested/null types) ────────────────────────────────────


def test_vendored_execute_with_nulls(con):
    """Vendored execute handles columns with null values."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    t = xo.memtable({"a": [1, None, 3], "b": ["x", None, "z"]})
    result = Backend.execute(con, t)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_vendored_execute_with_list_type(con):
    """Vendored execute handles nested (list) types via to_pylist()."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    with con._safe_raw_sql(
        "CREATE OR REPLACE TABLE test_nested AS SELECT [1, 2, 3] AS arr, 'a' AS label"
    ):
        pass
    try:
        t = con.table("test_nested")
        result = Backend.execute(con, t)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    finally:
        with con._safe_raw_sql('DROP TABLE IF EXISTS "test_nested"'):
            pass


# ── vendored to_pyarrow (column / scalar) ──────────────────────────────────


def test_vendored_to_pyarrow_column(con, batting):
    """Vendored to_pyarrow returns ChunkedArray for column expressions."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    col_expr = batting.limit(5).playerID
    result = Backend.to_pyarrow(con, col_expr)
    assert isinstance(result, pa.ChunkedArray)
    assert len(result) == 5


def test_vendored_to_pyarrow_scalar(con, batting):
    """Vendored to_pyarrow returns a scalar for scalar expressions."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    scalar_expr = batting.count()
    result = Backend.to_pyarrow(con, scalar_expr)
    assert isinstance(result, pa.Scalar) or isinstance(result, int)


# ── vendored to_pyarrow_batches ─────────────────────────────────────────────


def test_vendored_to_pyarrow_batches(con, batting):
    """Vendored to_pyarrow_batches returns RecordBatchReader."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    reader = Backend.to_pyarrow_batches(con, batting.limit(10))
    assert isinstance(reader, pa.ipc.RecordBatchReader)
    table = reader.read_all()
    assert table.num_rows == 10


# ── _normalize_arrow_schema ─────────────────────────────────────────────────


def test_normalize_arrow_schema_large_types(con):
    """_normalize_arrow_schema downcasts large_string to string."""
    table = pa.table(
        {
            "a": pa.array(["x", "y", "z"], type=pa.large_string()),
            "b": pa.array([b"a", b"b", b"c"], type=pa.large_binary()),
            "c": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    result = con._normalize_arrow_schema(table)
    assert result.schema.field("a").type == pa.string()
    assert result.schema.field("b").type == pa.binary()
    assert result.schema.field("c").type == pa.int64()


def test_normalize_arrow_schema_no_change(con):
    """_normalize_arrow_schema returns table unchanged when no large types."""
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = con._normalize_arrow_schema(table)
    assert result is table  # same object, no copy


# ── _register_in_memory_table (empty table) ─────────────────────────────────


def test_register_empty_memtable(con):
    """Empty memtable is created via DDL (ADBC ingest fails on zero rows)."""
    t = xo.memtable(
        pa.table(
            {"x": pa.array([], type=pa.int64()), "y": pa.array([], type=pa.string())}
        )
    )
    result = con.execute(t)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ── _get_schema_using_query ─────────────────────────────────────────────────


def test_get_schema_using_query(con):
    """_get_schema_using_query returns Schema from DESCRIBE."""
    schema = con._get_schema_using_query("SELECT 1 AS a, 'hello' AS b")
    assert "a" in schema
    assert "b" in schema


# ── _from_url ───────────────────────────────────────────────────────────────


def test_from_url_parsing(con):
    """_from_url parses URL components correctly."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    url = ParseResult(
        scheme="gizmosql",
        netloc="myuser:mypass@myhost:31337",
        path="/mydb/myschema",
        params="",
        query="useEncryption=true&disableCertificateVerification=true",
        fragment="",
    )

    # We can't actually connect to this URL, but we can test the parsing
    # by mocking connect
    parsed_kwargs = {}

    original_connect = Backend.connect

    def mock_connect(self, **kwargs):
        parsed_kwargs.update(kwargs)
        raise ConnectionError("mock")

    Backend.connect = mock_connect
    try:
        backend = Backend()
        with pytest.raises(ConnectionError, match="mock"):
            backend._from_url(
                url, useEncryption="true", disableCertificateVerification="true"
            )
        assert parsed_kwargs["host"] == "myhost"
        assert parsed_kwargs["port"] == 31337
        assert parsed_kwargs["user"] == "myuser"
        assert parsed_kwargs["password"] == "mypass"
        assert parsed_kwargs["database"] == "mydb"
        assert parsed_kwargs["schema"] == "myschema"
        assert parsed_kwargs["use_encryption"] is True
        assert parsed_kwargs["disable_certificate_verification"] is True
    finally:
        Backend.connect = original_connect


# ── _create_temp_view / _get_temp_view_definition ───────────────────────────


def test_create_temp_view(con):
    """_create_temp_view creates a temporary view."""
    view_name = "test_temp_view_cov"
    try:
        con._create_temp_view(view_name, "SELECT 42 AS answer")
        with con._safe_raw_sql(f'SELECT * FROM "{view_name}"') as cur:
            result = cur.fetch_arrow_table()
        assert result.column("answer")[0].as_py() == 42
    finally:
        with contextlib.suppress(Exception):
            with con._safe_raw_sql(f'DROP VIEW IF EXISTS "{view_name}"'):
                pass


# ── from_connection ─────────────────────────────────────────────────────────


def test_from_connection(con):
    """from_connection creates a new backend from an existing ADBC connection."""
    from xorq.vendor.ibis.backends.gizmosql import Backend

    new_backend = Backend.from_connection(con.con)
    assert new_backend.current_database == "main"
    tables = new_backend.list_tables()
    assert isinstance(tables, list)
