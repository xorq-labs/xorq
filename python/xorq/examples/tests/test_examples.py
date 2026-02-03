import pytest

import xorq.api as xo
from xorq.examples import Example, get_name_to_suffix
from xorq.examples.core import whitelist


def test_whitelist():
    assert [name in dir(xo.examples) for name in whitelist]


def test_attributes():
    for name in whitelist:
        example = getattr(xo.examples, name)
        assert isinstance(example, Example)

    with pytest.raises(AttributeError):
        xo.examples.missing


@pytest.mark.parametrize(
    "fetch",
    [
        pytest.param(
            lambda: xo.examples.penguins.fetch(xo.duckdb.connect(), table_name="t"),
            id="duckdb-name",
        ),
        pytest.param(
            lambda: xo.examples.penguins.fetch(table_name="t"), id="default-name"
        ),
        pytest.param(lambda: xo.examples.penguins.fetch(), id="default-no-name"),
        pytest.param(
            lambda: xo.examples.penguins.fetch(deferred=False), id="default-no-deferred"
        ),
    ],
)
def test_fetch_penguins(fetch):
    t = fetch()
    expected = t.execute()

    assert not expected.empty
    assert expected.shape


def test_parquet_examples():
    parquet_examples = tuple(
        name for name, suffix in get_name_to_suffix().items() if suffix == ".parquet"
    )
    assert len(parquet_examples) > 0

    assert not any(
        getattr(xo.examples, example_name)
        .fetch(xo.duckdb.connect(), table_name="t")
        .execute()
        .empty
        for example_name in parquet_examples
    )


def test_fetch_default_backend():
    batting = xo.examples.batting.fetch()
    awards_players = xo.examples.awards_players.fetch()
    assert batting._find_backend() is awards_players._find_backend()
    on = set(batting.columns).intersection(awards_players.columns)
    joined = batting.select(on).join(awards_players.select(on), predicates=on).execute()
    assert not joined.empty
