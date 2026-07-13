import pyarrow.parquet as pq
import pytest


ibis = pytest.importorskip("ibis")
# ibis.formats.pyarrow imports pyarrow_hotfix unconditionally, which upstream
# only declares under backend extras; skip the whole dir when it is absent
# rather than erroring at collection (these run in ci-test-ibis-compatibility)
pytest.importorskip("ibis.formats.pyarrow")


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
        {
            "a": "int64",
            "b": "string",
            "c": "float64",
            "d": "timestamp",
            "e": "date",
        },
        name="test_table",
    )
