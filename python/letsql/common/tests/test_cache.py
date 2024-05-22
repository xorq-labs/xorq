from pathlib import Path

import ibis
import pytest

from letsql.common.caching import ParquetCacheStorage


@pytest.fixture(scope="session")
def parquet_dir():
    root = Path(__file__).absolute().parents[4]
    data_dir = root / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


def test_put_get_drop(tmp_path, parquet_dir):
    batting_path = parquet_dir.joinpath("astronauts.parquet")

    con = ibis.datafusion.connect()
    t = con.register(batting_path, table_name="astronauts")

    storage = ParquetCacheStorage(path=tmp_path, source=con)
    put_node = storage.put(t, t.op())
    assert put_node is not None

    get_node = storage.get(t)
    assert get_node is not None

    storage.drop(t)
    with pytest.raises(KeyError):
        storage.get(t)
