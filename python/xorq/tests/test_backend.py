import pickle

import pytest

import xorq as xo


@pytest.mark.parametrize(
    "make_connection",
    (
        xo.postgres.connect_env,
        xo.postgres.connect_examples,
    ),
)
def test_remote_reconnect(make_connection):
    con = make_connection()
    expected = con.list_tables()
    actual = pickle.loads(pickle.dumps(con)).list_tables()
    assert expected  # smoke test that we have something meaningful to test
    assert actual == expected
