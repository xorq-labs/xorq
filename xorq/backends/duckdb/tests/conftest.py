import pytest

import xorq.api as xo


@pytest.fixture(scope="session")
def con(parquet_dir):
    conn = xo.connect()

    for name in ("astronauts", "functional_alltypes", "awards_players", "batting"):
        conn.read_parquet(parquet_dir / f"{name}.parquet", name)

    return conn


@pytest.fixture(scope="session")
def dirty_duckdb_con(csv_dir):
    conn = xo.duckdb.connect()
    conn.read_csv(csv_dir / "awards_players.csv", table_name="ddb_players")
    conn.read_csv(csv_dir / "batting.csv", table_name="batting")
    return conn


@pytest.fixture(scope="function")
def duckdb_con(dirty_duckdb_con):
    from duckdb import CatalogException

    expected_tables = ("ddb_players", "batting")
    for table in dirty_duckdb_con.list_tables():
        if table not in expected_tables:
            try:
                dirty_duckdb_con.drop_view(table, force=True)
            except CatalogException:
                dirty_duckdb_con.drop_table(table, force=True)
    yield dirty_duckdb_con


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def awards_players(con):
    return con.table("awards_players")


@pytest.fixture(scope="session")
def batting(con):
    return con.table("batting")
