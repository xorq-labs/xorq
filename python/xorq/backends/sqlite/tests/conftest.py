import pytest

import xorq.api as xo


@pytest.fixture(scope="session")
def sqlite_con():
    return xo.sqlite.connect()


@pytest.fixture(scope="session")
def astronauts_parquet_path(parquet_dir):
    return parquet_dir / "astronauts.parquet"


@pytest.fixture(scope="session")
def astronauts_csv_path(csv_dir):
    return csv_dir / "astronauts.csv"


@pytest.fixture(scope="session")
def persistent_sqlite_con(tmp_path_factory):
    path = tmp_path_factory.mktemp("database").joinpath("lite.db").resolve()
    return xo.sqlite.connect(str(path))
