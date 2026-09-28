"""`check-sources` stays read-only and dials each profile once (xorq-labs/xorq#2297).

The sqlite driver creates its database on open, so a probe that just connects
would write to the user's filesystem and misreport a deleted database as
`table-missing`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pyarrow as pa
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog import drift
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import open_con, sqlite_no_create
from xorq.vendor.ibis.backends.profiles import Profile


TABLE = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


@pytest.fixture
def world(tmp_path: Path, catalog_path: str) -> SimpleNamespace:
    """Two tables in one sqlite database, joined: two leaves on one profile."""
    db_path = tmp_path / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.create_table("u", TABLE.to_pandas())
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = catalog.add(con.table("t").join(con.table("u").view(), "a"))
    return SimpleNamespace(db_path=db_path, catalog_path=catalog_path, name=entry.name)


@pytest.fixture
def connects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every `Profile.get_con` call the command makes."""
    calls = []
    get_con = Profile.get_con

    def counted(self, *args, **kwargs):
        calls.append(self.con_name)
        return get_con(self, *args, **kwargs)

    monkeypatch.setattr(Profile, "get_con", counted)
    return calls


def stale_exists(monkeypatch: pytest.MonkeyPatch, path: Path) -> Callable:
    """Make the pre-check see ``path`` as present; returns the real ``exists``.

    Narrow on purpose: `Path.exists` carries the catalog's own resolution too.
    """
    exists = Path.exists
    monkeypatch.setattr(
        drift.Path, "exists", lambda self: True if self == path else exists(self)
    )
    return exists


def check_sources(runner: CliRunner, world: SimpleNamespace):
    return runner.invoke(
        cli, ["--path", world.catalog_path, "check-sources", world.name]
    )


def test_a_deleted_database_is_not_recreated(
    runner: CliRunner, world: SimpleNamespace, connects: list[str]
) -> None:
    world.db_path.unlink()

    result = check_sources(runner, world)
    assert result.exit_code == 2
    assert "unreachable" in result.output
    assert not world.db_path.exists()
    # The pre-check runs before any connect.
    assert connects == []


def test_a_corrupt_database_names_the_leaf(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    world.db_path.write_bytes(b"not a database")

    result = check_sources(runner, world)
    assert result.exit_code == 2
    assert "DatabaseTable t: unreachable" in result.output
    assert "file is not a database" in result.output
    assert "Traceback" not in result.output


def test_one_dead_profile_is_dialled_once(
    runner: CliRunner, world: SimpleNamespace, connects: list[str]
) -> None:
    """The file exists, so the pre-check passes and the driver fails instead."""
    world.db_path.write_bytes(b"not a database")

    result = check_sources(runner, world)
    assert result.output.count("unreachable") == 2
    assert connects == ["sqlite"]


def test_one_live_profile_is_dialled_once(
    runner: CliRunner, world: SimpleNamespace, connects: list[str]
) -> None:
    result = check_sources(runner, world)
    assert result.exit_code == 0
    assert result.output.count("equal") == 2
    assert connects == ["sqlite"]


# duckdb goes through `open_con`, not the command: a duckdb table serializes as
# a `Read` leaf, which `check-sources` only probes from xorq-labs/xorq#2296 on.


@pytest.mark.parametrize(
    "con_name, suffix", [("sqlite", ".sqlite"), ("duckdb", ".ddb")]
)
def test_a_missing_database_is_never_created(
    tmp_path: Path, con_name: str, suffix: str
) -> None:
    db_path = tmp_path / f"gone{suffix}"
    profile = Profile(con_name=con_name, kwargs_tuple=(("database", str(db_path)),))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_con(profile, {})
    assert not db_path.exists()


@pytest.mark.parametrize(
    "con_name, suffix", [("sqlite", ".sqlite"), ("duckdb", ".ddb")]
)
def test_the_driver_refuses_to_create_when_the_pre_check_goes_stale_either_way(
    tmp_path: Path, con_name: str, suffix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pre-check neutered, the driver is the one that has to refuse."""
    db_path = tmp_path / f"gone{suffix}"
    exists = stale_exists(monkeypatch, db_path)
    profile = Profile(con_name=con_name, kwargs_tuple=(("database", str(db_path)),))

    with pytest.raises(Exception, match="unable to open|does not exist"):
        open_con(profile, {})
    assert not exists(db_path)


def test_the_driver_refuses_to_create_when_the_pre_check_goes_stale(
    runner: CliRunner,
    world: SimpleNamespace,
    connects: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file can go between the check and the connect; the driver refuses."""
    world.db_path.unlink()
    exists = stale_exists(monkeypatch, world.db_path)

    result = check_sources(runner, world)
    assert result.exit_code == 2
    assert "unreachable" in result.output
    assert not exists(world.db_path)
    # Dialled this time, and the driver declined to create.
    assert connects == ["sqlite"]


def test_a_question_mark_in_the_path_stays_in_the_path(tmp_path: Path) -> None:
    """Unescaped, sqlite cuts at the `?`: `rwc` creates the truncated path."""
    db_path = tmp_path / "we?ird.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.disconnect()
    profile = Profile(con_name="sqlite", kwargs_tuple=(("database", str(db_path)),))

    _, kwargs = sqlite_no_create(profile)

    assert profile.get_con(**kwargs).list_tables() == ["t"]
    assert not (tmp_path / "we").exists()


def test_a_read_only_duckdb_connection_cannot_write(tmp_path: Path) -> None:
    """The duckdb flag blocks writes for the whole session."""
    db_path = tmp_path / "live.ddb"
    con = xo.duckdb.connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.disconnect()
    profile = Profile(con_name="duckdb", kwargs_tuple=(("database", str(db_path)),))

    con = drift.connect(profile)
    assert con.list_tables() == ["t"]
    with pytest.raises(Exception, match="read-only|Cannot execute"):
        con.create_table("u", TABLE.to_pandas())


@pytest.mark.parametrize(
    ("con_name", "kwargs_tuple"),
    [
        pytest.param("duckdb", (("database", ":memory:"),), id="duckdb-in-memory"),
        pytest.param(
            "duckdb", (("database", ":memory:scratch"),), id="duckdb-named-in-memory"
        ),
        pytest.param("sqlite", (("database", None),), id="sqlite-in-memory"),
        pytest.param(
            "sqlite", (("database", ":memory:"),), id="sqlite-in-memory-spelled-out"
        ),
        pytest.param("duckdb", (("database", "md:analytics"),), id="motherduck-handle"),
        pytest.param(
            "sqlite",
            (("database", "file:/tmp/x.sqlite?mode=ro"), ("uri", True)),
            id="sqlite-uri-with-a-mode",
        ),
        pytest.param("duckdb", (), id="duckdb-no-database-recorded"),
        pytest.param("postgres", (("database", "analytics"),), id="not-file-backed"),
    ],
)
def test_a_target_that_is_no_local_file_is_left_alone(
    con_name: str, kwargs_tuple: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing reaches the driver for a target it would not create.

    Asserted there rather than on a predicate: a target that is no file and a
    URI that already refuses are exempt by different routes.
    """
    seen = []
    monkeypatch.setattr(Profile, "get_con", lambda self, **kwargs: seen.append(kwargs))
    profile = Profile(con_name=con_name, kwargs_tuple=kwargs_tuple)

    drift.connect(profile)

    assert seen == [{}]


@pytest.mark.parametrize(
    "mode_part",
    [pytest.param("", id="no-mode"), pytest.param("?mode=rwc", id="mode-rwc")],
)
def test_a_recorded_uri_that_would_create_gets_a_mode(
    tmp_path: Path, mode_part: str
) -> None:
    """No `mode=` means `rwc`: being a URI is not itself a refusal to create."""
    db_path = tmp_path / "gone.sqlite"
    profile = Profile(
        con_name="sqlite",
        kwargs_tuple=(("database", f"file:{db_path}{mode_part}"), ("uri", True)),
    )

    assert sqlite_no_create(profile) == (
        # No path to check: a recorded URI is left to the driver.
        None,
        {"database": f"file:{db_path}?mode=rw", "uri": True},
    )
    with pytest.raises(Exception, match="unable to open"):
        open_con(profile, {})
    assert not db_path.exists()


def test_a_file_string_without_uri_is_an_ordinary_path(tmp_path: Path) -> None:
    """Without `uri=True` the whole string is a filename, and `rwc` creates it."""
    target = f"file:{tmp_path / 'gone.sqlite'}?mode=ro"
    profile = Profile(con_name="sqlite", kwargs_tuple=(("database", target),))

    assert sqlite_no_create(profile) == (
        target,
        {"database": f"file:{quote(target)}?mode=rw", "uri": True},
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_con(profile, {})
    assert not Path(target).exists()


def test_a_doubled_leading_slash_is_not_an_authority(tmp_path: Path) -> None:
    """sqlite reads what follows `file://` up to the next `/` as an authority,
    and takes only an empty one or `localhost`, so an absolute path needs the
    empty authority spelled out."""
    db_path = tmp_path / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.disconnect()
    # A valid spelling of the same path, and the one that would lose its head.
    profile = Profile(con_name="sqlite", kwargs_tuple=(("database", f"/{db_path}"),))

    assert open_con(profile, {}).list_tables() == ["t"]


def test_a_recorded_uri_still_reaches_its_tables(tmp_path: Path) -> None:
    """Rewriting the mode leaves the rest of the URI, and its path, in place."""
    db_path = tmp_path / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.disconnect()
    profile = Profile(
        con_name="sqlite",
        kwargs_tuple=(("database", f"file:{db_path}"), ("uri", True)),
    )

    assert open_con(profile, {}).list_tables() == ["t"]
