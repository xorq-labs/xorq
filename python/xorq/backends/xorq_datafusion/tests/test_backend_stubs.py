from __future__ import annotations

import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.common.exceptions as com


@pytest.fixture
def stub_con():
    return xo.connect()


@pytest.fixture
def stub_con_with_table(stub_con):
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    stub_con.register(table, table_name="t")
    return stub_con


def test_current_catalog_not_implemented(stub_con):
    with pytest.raises(NotImplementedError):
        stub_con.current_catalog


def test_current_database_not_implemented(stub_con):
    with pytest.raises(NotImplementedError):
        stub_con.current_database


def test_drop_catalog_unsupported(stub_con):
    with pytest.raises(com.UnsupportedOperationError):
        stub_con.drop_catalog("nonexistent")


def test_disconnect_noop(stub_con):
    assert stub_con.disconnect() is None


def test_to_parquet_roundtrip(stub_con_with_table, tmp_path):
    expr = stub_con_with_table.table("t")
    out = tmp_path / "out.parquet"
    stub_con_with_table.to_parquet(expr, out)

    assert out.exists()
    written = pa.parquet.read_table(out)
    assert written.num_rows == 3
    assert set(written.column_names) == {"a", "b"}


def test_extract_catalog(stub_con_with_table):
    catalog = stub_con_with_table._extract_catalog("SELECT * FROM t")

    assert set(catalog) == {"t"}
    assert catalog["t"].columns == ["a", "b"]
