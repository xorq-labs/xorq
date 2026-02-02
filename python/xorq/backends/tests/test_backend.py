import operator

import pytest

import xorq.api as xo
import xorq.expr.relations as rel
from xorq.common.utils.graph_utils import (
    find_all_sources,
    walk_nodes,
)


pytest.importorskip("psycopg")


def get_method(connect_parts):
    (backend, method) = connect_parts
    connect = getattr(getattr(xo, backend), method)
    return connect


@pytest.mark.parametrize(
    "connect_parts",
    (
        ("duckdb", "connect"),
        ("postgres", "connect_env"),
        ("datafusion", "connect"),
        (
            "xorq",
            "connect",
        ),
        ("pandas", "connect"),
        ("sqlite", "connect"),
        pytest.param(
            ("snowflake", "connect_env_keypair"),
            marks=[pytest.mark.snowflake, pytest.mark.slow],
        ),
    ),
)
def test_con_equality(connect_parts):
    # where do we want to the connection to be the same?
    n = 3
    connect = get_method(connect_parts)
    cons = set(connect() for _ in range(3))
    assert len(cons) == n


@pytest.mark.parametrize(
    "connect_parts",
    (
        ("duckdb", "connect"),
        ("postgres", "connect_env"),
        (
            "xorq",
            "connect",
        ),
        ("pandas", "connect"),
        ("sqlite", "connect"),
        pytest.param(
            ("snowflake", "connect_env_keypair"),
            marks=[pytest.mark.snowflake, pytest.mark.slow],
        ),
    ),
)
def test_con_equality_read(connect_parts, parquet_dir):
    connect = get_method(connect_parts)
    on = "playerID"
    ts = t0, t1 = tuple(
        xo.deferred_read_parquet(parquet_dir.joinpath("batting.parquet"), connect())
        .filter(operator.eq(xo._.yearID, year))
        .select(on)
        for year in (2014, 2015)
    )
    joined = t0.join(t1.into_backend(t0._find_backend()), on)
    assert not joined.execute().empty
    actual = find_all_sources(joined)
    expected = tuple(t._find_backend() for t in ts)
    assert actual == expected

    (r0, *rest) = walk_nodes(rel.Read, joined)
    assert r0 and rest
