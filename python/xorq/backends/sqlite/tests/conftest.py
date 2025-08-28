import pytest

import xorq.api as xo


@pytest.fixture(scope="session")
def sqlite_con():
    return xo.sqlite.connect()


@pytest.fixture(scope="session")
def astronauts_parquet_path(parquet_dir):
    return parquet_dir / "astronauts.parquet"
