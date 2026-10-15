import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import xorq.api as xo
from xorq.api import _
from xorq.expr.builders import _FROM_TAG_NODE_REGISTRY


@pytest.fixture
def saved_registry():
    """Snapshot the builder ``TagHandler`` registry around a test.

    Tests that call ``register_tag_handler`` mutate process-global state in
    ``xorq.expr.builders``; this fixture restores that state afterward so
    tests don't leak handlers into each other or into builtin lookups.
    """
    import xorq.expr.builders as _builders_mod  # noqa: PLC0415

    saved = dict(_FROM_TAG_NODE_REGISTRY)
    saved_keys = _builders_mod._BUILTIN_KEYS
    saved_init = _builders_mod._initialized
    yield
    _FROM_TAG_NODE_REGISTRY.clear()
    _FROM_TAG_NODE_REGISTRY.update(saved)
    _builders_mod._BUILTIN_KEYS = saved_keys
    _builders_mod._initialized = saved_init


array_types_df = pd.DataFrame(
    [
        (
            [np.int64(1), 2, 3],
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            "a",
            1.0,
            [[], [np.int64(1), 2, 3], None],
        ),
        (
            [4, 5],
            ["d", "e"],
            [4.0, 5.0],
            "a",
            2.0,
            [],
        ),
        (
            [6, None],
            ["f", None],
            [6.0, np.nan],
            "a",
            3.0,
            [None, [], None],
        ),
        (
            [None, 1, None],
            [None, "a", None],
            [],
            "b",
            4.0,
            [[1], [2], [], [3, 4, 5]],
        ),
        (
            [2, None, 3],
            ["b", None, "c"],
            np.nan,
            "b",
            5.0,
            None,
        ),
        (
            [4, None, None, 5],
            ["d", None, None, "e"],
            [4.0, np.nan, np.nan, 5.0],
            "c",
            6.0,
            [[1, 2, 3]],
        ),
    ],
    columns=[
        "x",
        "y",
        "z",
        "grouper",
        "scalar_column",
        "multi_dim",
    ],
)

win = pd.DataFrame(
    {
        "g": ["a", "a", "a", "a", "a"],
        "x": [0, 1, 2, 3, 4],
        "y": [3, 2, 0, 1, 1],
    }
)

expected_tables = (
    "array_types",
    "astronauts",
    "awards_players",
    "awards_players_special_types",
    "batting",
    "diamonds",
    "functional_alltypes",
    "geo",
    "geography_columns",
    "geometry_columns",
    "json_t",
    "map",
    "spatial_ref_sys",
    "topk",
    "tzone",
)


def remove_unexpected_tables(dirty):
    # drop tables
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_table(table, force=True)

    # drop view
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_view(table, force=True)

    actual = sorted(dirty.list_tables())
    expected = sorted(expected_tables)
    if actual != expected:
        missing = tuple(t for t in expected if t not in actual)
        extra = tuple(t for t in actual if t not in expected)
        raise ValueError(
            {
                "missing": missing,
                "extra": extra,
            }
        )


@pytest.fixture(scope="function")
def pg():
    conn = xo.postgres.connect_env()
    remove_unexpected_tables(conn)
    yield conn
    remove_unexpected_tables(conn)


@pytest.fixture(scope="session")
def root_dir():
    return Path(__file__).absolute().parents[2]


@pytest.fixture(scope="session")
def parquet_dir(root_dir):
    data_dir = root_dir / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


@pytest.fixture(scope="session")
def fixture_dir(root_dir):
    return root_dir.joinpath("python", "xorq", "tests", "fixtures")


@pytest.fixture(scope="session")
def data_dir(root_dir):
    data_dir = root_dir / "ci" / "ibis-testing-data"
    return data_dir


@pytest.fixture(scope="session")
def csv_dir(data_dir):
    csv_dir = data_dir / "csv"
    return csv_dir


@pytest.fixture(scope="session")
def examples_dir(root_dir):
    examples_dir = root_dir / "examples"
    return examples_dir


@pytest.fixture
def union_subsets(alltypes, alltypes_df):
    """Three overlapping id-range slices of alltypes, each with its columns in a
    different order, as expressions and as the matching pandas frames."""
    cols_a, cols_b, cols_c = (alltypes.columns.copy() for _ in range(3))

    random.seed(89)
    random.shuffle(cols_a)
    random.shuffle(cols_b)
    random.shuffle(cols_c)
    assert cols_a != cols_b != cols_c

    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))[cols_a]
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))[cols_b]
    c = alltypes.filter((_.id >= 5213) & (_.id <= 5220))[cols_c]

    da = alltypes_df[(alltypes_df.id >= 5200) & (alltypes_df.id <= 5210)][cols_a]
    db = alltypes_df[(alltypes_df.id >= 5205) & (alltypes_df.id <= 5215)][cols_b]
    dc = alltypes_df[(alltypes_df.id >= 5213) & (alltypes_df.id <= 5220)][cols_c]

    return (a, b, c), (da, db, dc)


@pytest.fixture(scope="module")
def time_keyed_df1():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", periods=6)
            ),
            "key": [1, 2, 3, 1, 2, 3],
            "value": [1.2, 1.4, 2.0, 4.0, 8.0, 16.0],
        }
    )


@pytest.fixture(scope="module")
def time_keyed_df2():
    return pd.DataFrame(
        {
            "time": pd.Series(
                pd.date_range(start="2017-01-02 01:02:03.234", freq="3D", periods=3)
            ),
            "key": [1, 2, 3],
            "other_value": [1.1, 1.2, 2.2],
        }
    )


@pytest.fixture(scope="session")
def ddl_file(root_dir):
    return root_dir / "db" / "datafusion.sql"


def statements(ddl_file: Path):
    return (
        statement
        for statement in map(str.strip, ddl_file.read_text().split(";"))
        if statement
    )


def make_xo_con(parquet_dir, ddl_file):
    """A xorq_datafusion connection over the ibis test data.

    The root `xo_con` is one of these; a test directory that mutates its
    tables overrides `xo_con` with its own so nothing leaks across directories.
    The table fixtures below are function-scoped on purpose: a session-scoped
    one would be cached once and bind every directory to the first
    connection that requested it."""
    conn = xo.connect()
    for name in (
        "astronauts",
        "awards_players",
        "batting",
        "diamonds",
        "functional_alltypes",
    ):
        conn.read_parquet(parquet_dir / f"{name}.parquet", name)
    conn.create_table("array_types", array_types_df)
    conn.create_table("win", win)
    if ddl_file.is_file():
        for statement in statements(ddl_file):
            with conn._safe_raw_sql(statement):  # noqa
                pass
    return conn


@pytest.fixture(scope="session")
def xo_con(parquet_dir, ddl_file):
    return make_xo_con(parquet_dir, ddl_file)


@pytest.fixture
def alltypes(xo_con):
    return xo_con.table("functional_alltypes")


@pytest.fixture
def functional_alltypes(alltypes):
    return alltypes


@pytest.fixture
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture
def sorted_df(alltypes_df):
    return alltypes_df.sort_values("id").reset_index(drop=True)


@pytest.fixture
def batting(xo_con):
    return xo_con.table("batting")


@pytest.fixture
def batting_df(batting):
    return batting.execute()


@pytest.fixture
def awards_players(xo_con):
    return xo_con.table("awards_players")


@pytest.fixture
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)


@pytest.fixture
def diamonds(xo_con):
    return xo_con.table("diamonds")


@pytest.fixture
def array_types(xo_con):
    return xo_con.table("array_types")


@pytest.fixture
def struct(xo_con):
    return xo_con.table("structs")


@pytest.fixture
def struct_df(struct):
    return struct.execute()


@pytest.fixture
def win_table(xo_con):
    return xo_con.table("win")
