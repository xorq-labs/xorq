import pytest

import xorq.api as xo


@pytest.fixture(scope="session")
def sqlite_con():
    return xo.sqlite.connect()
