import pickle

import pytest

import xorq.api as xo


@pytest.mark.parametrize(
    "make_connection",
    (
        pytest.param(lambda: xo.postgres.connect_env(), id="env"),
        pytest.param(
            lambda: xo.postgres.connect_examples(),
            id="examples",
            marks=pytest.mark.examples_remote,
        ),
    ),
)
@pytest.mark.slow(level=1)
def test_remote_reconnect(make_connection):
    con = make_connection()
    expected = con.list_tables()
    actual = pickle.loads(pickle.dumps(con)).list_tables()
    assert expected  # smoke test that we have something meaningful to test
    assert actual == expected
