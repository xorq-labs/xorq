import pyarrow.parquet as pq
import pytest


ibis = pytest.importorskip("ibis")


@pytest.fixture(scope="session")
def ibis_con(parquet_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "small_alltypes.parquet"
    table = pq.read_table(parquet_dir / "functional_alltypes.parquet").slice(length=100)
    pq.write_table(table, path)

    conn = ibis.duckdb.connect()
    conn.read_parquet(path, table_name="alltypes")
    conn.read_parquet(parquet_dir / "astronauts.parquet", table_name="astronauts")
    return conn


@pytest.fixture(scope="session")
def ibis_alltypes(ibis_con):
    return ibis_con.table("alltypes")


@pytest.fixture(scope="session")
def ibis_astronauts(ibis_con):
    return ibis_con.table("astronauts")


@pytest.fixture
def t():
    return ibis.table(
        dict(
            a="int64",
            b="string",
            c="float64",
            d="timestamp",
            e="date",
        ),
        name="test_table",
    )
