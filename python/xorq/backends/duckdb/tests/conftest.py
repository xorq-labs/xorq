import pytest
from duckdb import CatalogException

import xorq.api as xo
from xorq.conftest import make_xo_con


@pytest.fixture(scope="session")
def xo_con(parquet_dir, ddl_file):
    # this directory's own connection: its tests create and drop tables
    return make_xo_con(parquet_dir, ddl_file)


@pytest.fixture(scope="session")
def con(xo_con):
    # historically xo.connect() over the test parquet, not a duckdb backend
    return xo_con


@pytest.fixture(scope="session")
def dirty_duckdb_con(csv_dir):
    conn = xo.duckdb.connect()
    conn.read_csv(csv_dir / "awards_players.csv", table_name="ddb_players")
    conn.read_csv(csv_dir / "batting.csv", table_name="batting")
    return conn


@pytest.fixture(scope="function")
def duckdb_con(dirty_duckdb_con):
    expected_tables = ("ddb_players", "batting")
    for table in dirty_duckdb_con.list_tables():
        if table not in expected_tables:
            try:
                dirty_duckdb_con.drop_view(table, force=True)
            except CatalogException:
                dirty_duckdb_con.drop_table(table, force=True)
    yield dirty_duckdb_con
