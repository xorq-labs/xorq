from pathlib import Path

import pytest

import xorq.api as xo


@pytest.fixture(scope="session")
def sqlite_con():
    return xo.sqlite.connect()


@pytest.fixture(scope="session")
def astronauts_parquet_path(parquet_dir):
    return parquet_dir / "astronauts.parquet"


@pytest.fixture(scope="session")
def persistent_sqlite_con():
    path = Path(__file__).parent.joinpath("lite.db").resolve()
    return xo.sqlite.connect(str(path))
