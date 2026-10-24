import os
import random
import re
import shutil
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

import xorq.api as xo
import xorq.config
from xorq.api import _
from xorq.expr.builders import _FROM_TAG_NODE_REGISTRY


# one Hypothesis profile, locally and on CI: 75 examples per property, no
# deadline (backend execution time varies with load)
settings.register_profile("xorq", max_examples=75, deadline=None)
settings.load_profile("xorq")


_SESSION_CLEANUP_ERROR = pytest.StashKey[str | None]()
_CLEANUP_SECTION = "tmp_path cleanup"


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Remove the session's temp dir after a green run, as pytest itself does.

    The same rule as pytest's own hook, which runs after this one and then
    finds nothing to remove: policy "failed", exit status 0, and a basetemp
    pytest chose. A ``--basetemp`` the user gave is kept, and so is each xdist
    worker's, which xdist passes as ``--basetemp``: the controller removes the
    whole tree once every worker is done. pytest's removal ignores errors and
    so leaves git-annex's read-only dirs behind; this one makes them
    removable first and reports what it still cannot remove.
    """
    factory = session.config._tmp_path_factory
    if (
        factory._basetemp is not None
        and factory._given_basetemp is None
        and session.config.getini("tmp_path_retention_policy") == "failed"
        and exitstatus == 0
    ):
        session.config.stash[_SESSION_CLEANUP_ERROR] = _remove(factory._basetemp)


@pytest.fixture(scope="session", autouse=True)
def _scratch_env(tmp_path_factory):
    """Keep what tests' git and ``xorq`` subprocesses write inside the session.

    ``git init`` copies its sample hooks into every repo the catalog tests make
    (~1,800 a run), and each ``xorq run`` subprocess leaves a run log under
    ``~/.local/share/xorq/runs``. An empty template dir and a session runs dir
    stop both: the env var reaches subprocesses, and in-process runs read
    ``env_config``, which was built from the environment at import.
    """
    runs_dir = str(tmp_path_factory.mktemp("runs"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_TEMPLATE_DIR", str(tmp_path_factory.mktemp("git-template")))
        mp.setenv("XORQ_RUNS_LOGS_DIR", runs_dir)
        mp.setattr(
            "xorq.config.env_config",
            xorq.config.env_config.clone(XORQ_RUNS_LOGS_DIR=runs_dir),
        )
        yield


_OUTCOMES = pytest.StashKey[dict[str, str]]()
_TMP_PATH = pytest.StashKey[Path]()


def _make_removable(path):
    # removal needs read, write and search on each dir on POSIX (git-annex
    # makes its object dirs read-only, a test may leave one unreadable) and no
    # read-only file on Windows (git makes its objects read-only): the tree is
    # going, so give the owner everything, and shutil.rmtree meets no
    # permission error of ours; a symlink's chmod would reach its target
    path.chmod(stat.S_IRWXU)
    with os.scandir(path) as scanned:
        entries = tuple(scanned)
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            _make_removable(Path(entry.path))
        elif not entry.is_symlink():
            os.chmod(entry.path, stat.S_IRUSR | stat.S_IWUSR)


def _remove(path, recreate=False):
    """Remove path; return why it could not be, else None.

    What is left is a leaked writer or a dir we do not own: worth reporting,
    not a reason to fail a test or the run.
    """
    try:
        _make_removable(path)
        shutil.rmtree(path)
        if recreate:
            path.mkdir()
    except OSError as e:
        return f"{path}: {e!r}"
    return None


def _keeps_tmp_path(outcomes, policy):
    # pytest keys "failed" on the call passing; a failed setup or teardown
    # keeps the dir too, and an xfailed or skipped call keeps it as in pytest
    match policy:
        case "all":
            return True
        case "none":
            return False
        case "failed":
            return "failed" in outcomes.values() or (
                outcomes.get("call", "passed") != "passed"
            )


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    """pytest's ``tmp_path``, kept by ``tmp_path_retention_policy`` our way.

    pytest's own fixture, under "failed", keeps a dir only when the call did
    not pass, so a test that errored in setup or teardown loses it; removes
    the rest with ``rmtree(ignore_errors=True)``, which skips git-annex's
    read-only dirs; and frees a removed dir's name for a later test. Here a
    test whose call did not pass, or that errored, keeps its files, read-only
    modes included, and any other test's dir is emptied, not removed, so its
    name stays taken. "all" keeps every dir and "none" empties every dir, as
    in pytest. What cannot be removed is listed in the terminal summary.
    """
    # same name as pytest's own
    path = tmp_path_factory.mktemp(
        re.sub(r"[\W]", "_", request.node.name)[:30], numbered=True
    )
    request.node.stash[_TMP_PATH] = path
    return path


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    # applies tmp_path's policy at the teardown report, the first point that
    # knows whether a finalizer failed, a session fixture's included; the
    # report's outcome is the test's, never the cleanup's
    report = yield
    outcomes = item.stash.setdefault(_OUTCOMES, {})
    outcomes[report.when] = report.outcome
    path = item.stash.get(_TMP_PATH, None)
    policy = item.config.getini("tmp_path_retention_policy")
    if (
        report.when == "teardown"
        and path is not None
        and not _keeps_tmp_path(outcomes, policy)
        and (error := _remove(path, recreate=True)) is not None
    ):
        # a section reaches the xdist controller with the report
        report.sections.append((_CLEANUP_SECTION, error))
    return report


def pytest_terminal_summary(terminalreporter, config):
    # the reporter holds every report, a worker's under xdist included
    errors = (
        *(
            error
            for reports in terminalreporter.stats.values()
            for report in reports
            for name, error in getattr(report, "sections", ())
            if name == _CLEANUP_SECTION
        ),
        *filter(None, [config.stash.get(_SESSION_CLEANUP_ERROR, None)]),
    )
    if errors:
        terminalreporter.write_sep("=", "tmp_path cleanup errors", yellow=True)
        for error in errors:
            terminalreporter.line(error)


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

    The root `xo_con` is one of these and is shared by every directory that
    does not override it; a directory whose tests create, register or cache
    tables on `xo_con` overrides it with its own `make_xo_con` connection so
    nothing leaks across directories.
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
    # register, not create_table: create_table leaves an ibis_*_memtable_<hash>
    # registration behind for every frame it materializes
    conn.register(array_types_df, "array_types")
    conn.register(win, "win")
    if ddl_file.is_file():
        for statement in statements(ddl_file):
            with conn._safe_raw_sql(statement):  # noqa
                pass
    return conn


@pytest.fixture(scope="session")
def xo_con(parquet_dir, ddl_file):
    return make_xo_con(parquet_dir, ddl_file)


@pytest.fixture(scope="session")
def executed_frames():
    """Executed frames keyed by (connection, table): the function-scoped
    `*_df` fixtures below read each table once per directory connection."""
    return {}


def executed(frames, table):
    key = (id(table._find_backend()), table.get_name())
    if key not in frames:
        frames[key] = table.execute()
    return frames[key].copy()


@pytest.fixture
def alltypes(xo_con):
    return xo_con.table("functional_alltypes")


@pytest.fixture
def alltypes_df(alltypes, executed_frames):
    return executed(executed_frames, alltypes)


@pytest.fixture
def sorted_df(alltypes_df):
    return alltypes_df.sort_values("id").reset_index(drop=True)


@pytest.fixture
def batting(xo_con):
    return xo_con.table("batting")


@pytest.fixture
def batting_df(batting, executed_frames):
    return executed(executed_frames, batting)


@pytest.fixture
def awards_players(xo_con):
    return xo_con.table("awards_players")


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
def struct_df(struct, executed_frames):
    return executed(executed_frames, struct)


@pytest.fixture
def win_table(xo_con):
    return xo_con.table("win")
