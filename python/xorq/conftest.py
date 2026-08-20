import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import xorq.api as xo
from xorq.expr.builders import _FROM_TAG_NODE_REGISTRY


@pytest.fixture(autouse=True)
def _strict_monkeypatch(monkeypatch):
    """Fail fast when a string-target monkeypatch names a non-existent module attribute.

    Catches the common mistake of patching ``"xorq.a.b.name"`` when ``name``
    was imported into ``xorq.a.b`` at the top level but has since been deferred
    — meaning the attribute no longer exists on the module and the patch would
    silently miss.
    """
    _original = type(monkeypatch).setattr

    def _checked_setattr(self, target, *args, **kwargs):
        if isinstance(target, str):
            mod_path, _, attr = target.rpartition(".")
            if mod_path:
                mod = importlib.import_module(mod_path)
                if not hasattr(mod, attr):
                    raise AttributeError(
                        f"{attr!r} is not an attribute of {mod_path!r}; "
                        f"patch the canonical source module instead"
                    )
        return _original(self, target, *args, **kwargs)

    monkeypatch.setattr = _checked_setattr.__get__(monkeypatch)


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
