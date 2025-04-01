import pyarrow as pa
import pytest

import xorq as xo
from xorq.backends.let.datafusion.provider import IbisTableProvider


@pytest.fixture
def tmp_model_dir(tmpdir):
    # Create a temporary directory for the model
    model_dir = tmpdir.mkdir("models")
    return model_dir


@pytest.fixture(scope="session")
def con(data_dir):
    conn = xo.connect()
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
