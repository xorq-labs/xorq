import pickle

import pytest

import xorq.api as xo


@pytest.mark.parametrize(
    "make_connection",
    (
        "connect_env",
        "connect_examples",
    ),
)
@pytest.mark.slow(level=1)
def test_remote_reconnect(make_connection):
    con = getattr(xo.postgres, make_connection)()
    expected = con.list_tables()
    actual = pickle.loads(pickle.dumps(con)).list_tables()
    assert expected  # smoke test that we have something meaningful to test
    assert actual == expected
