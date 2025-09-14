import pytest

import xorq.api as xo


@pytest.fixture(scope="function")
def con():
    return xo.connect()
