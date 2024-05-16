from pathlib import Path

import pyarrow as pa
import pytest

import letsql as ls
from letsql.backends.datafusion.provider import IbisTableProvider


@pytest.fixture(scope="session")
def data_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data"
    return data_dir


@pytest.fixture(scope="session")
def con(data_dir):
    conn = ls.connect()
    parquet_dir = data_dir / "parquet"
    conn.register(parquet_dir / "functional_alltypes.parquet", "functional_alltypes")

    return conn


def test_table_provider_scan(con):
    table_provider = IbisTableProvider(con.table("functional_alltypes"))
    batches = table_provider.scan()

    assert batches is not None
    assert isinstance(batches, pa.RecordBatchReader)


def test_table_provider_schema(con):
    table_provider = IbisTableProvider(con.table("functional_alltypes"))
    schema = table_provider.schema()
    assert schema is not None
    assert isinstance(schema, pa.Schema)
