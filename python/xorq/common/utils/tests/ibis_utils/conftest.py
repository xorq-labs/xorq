import ibis
import pytest


@pytest.fixture(scope="session")
def ibis_con(parquet_dir):
    conn = ibis.duckdb.connect()
    conn.read_parquet(
        parquet_dir / "functional_alltypes.parquet", table_name="alltypes"
    )
    conn.read_parquet(parquet_dir / "astronauts.parquet", table_name="astronauts")
    return conn


@pytest.fixture(scope="session")
def ibis_alltypes(ibis_con):
    return ibis_con.table("alltypes")


@pytest.fixture(scope="session")
def ibis_astronauts(ibis_con):
    return ibis_con.table("astronauts")
