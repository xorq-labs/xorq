"""`check-sources` stays read-only and dials each profile once (xorq-labs/xorq#2297).

The sqlite driver creates its database file on open, so a probe that just
connects would both write to the user's filesystem and misreport a deleted
database as `table-missing`: the fresh empty file lists no tables.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog import drift
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import (
    missing_database_file,
    no_create_kwargs,
    open_con,
)
from xorq.vendor.ibis.backends.profiles import Profile


TABLE = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


@pytest.fixture
def world(tmp_path: Path, catalog_path: str) -> SimpleNamespace:
    """One sqlite database holding two tables, joined into one entry.

    Two leaves on one profile is what the connection cache has to collapse.
    """
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
    # The pre-check runs before any connect, not inside the failure handler.
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
    """The file exists, so the pre-check passes and the driver is the one to fail."""
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


# duckdb is exercised through `open_con` rather than the command: a duckdb table
# serializes as a `Read` leaf, which `check-sources` only probes from
# xorq-labs/xorq#2296 on. The connection policy under test is the same one.


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
    monkeypatch.setattr(drift, "missing_database_file", lambda profile: None)
    db_path = tmp_path / f"gone{suffix}"
    profile = Profile(con_name=con_name, kwargs_tuple=(("database", str(db_path)),))

    with pytest.raises(Exception, match="unable to open|does not exist"):
        open_con(profile, {})
    assert not db_path.exists()


def test_the_driver_refuses_to_create_when_the_pre_check_goes_stale(
    runner: CliRunner,
    world: SimpleNamespace,
    connects: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file can go between the check and the connect, and only the driver
    can close that window. With the pre-check neutered, the connect is the one
    that has to refuse."""
    monkeypatch.setattr(drift, "missing_database_file", lambda profile: None)
    world.db_path.unlink()

    result = check_sources(runner, world)
    assert result.exit_code == 2
    assert "unreachable" in result.output
    assert not world.db_path.exists()
    # The driver was dialled this time, and declined to create the database.
    assert connects == ["sqlite"]


def test_a_question_mark_in_the_path_stays_in_the_path(tmp_path: Path) -> None:
    """sqlite splits a URI at the first `?`. Unescaped, the path truncates, the
    mode vanishes with the rest of the garbled query, and the default `rwc`
    creates a database at the truncated path."""
    db_path = tmp_path / "we?ird.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.disconnect()
    profile = Profile(con_name="sqlite", kwargs_tuple=(("database", str(db_path)),))

    assert profile.get_con(**no_create_kwargs(profile)).list_tables() == ["t"]
    assert not (tmp_path / "we").exists()


def test_a_read_only_duckdb_connection_cannot_write(tmp_path: Path) -> None:
    """The duckdb flag blocks writes for the whole session, which is the
    stronger half of what a read-only command wants from it."""
    db_path = tmp_path / "live.ddb"
    con = xo.duckdb.connect(str(db_path))
    con.create_table("t", TABLE.to_pandas())
    con.disconnect()
    profile = Profile(con_name="duckdb", kwargs_tuple=(("database", str(db_path)),))

    con = profile.get_con(**no_create_kwargs(profile))
    assert con.list_tables() == ["t"]
    with pytest.raises(Exception, match="read-only|Cannot execute"):
        con.create_table("u", TABLE.to_pandas())


@pytest.mark.parametrize(
    ("con_name", "target"),
    [
        pytest.param("duckdb", ":memory:", id="duckdb-in-memory"),
        pytest.param("sqlite", None, id="sqlite-in-memory"),
        pytest.param("duckdb", "md:analytics", id="motherduck-handle"),
        pytest.param("sqlite", "file:/tmp/x.sqlite?mode=ro", id="sqlite-uri"),
        pytest.param("postgres", "analytics", id="not-file-backed"),
    ],
)
def test_a_target_that_is_no_local_file_is_left_alone(
    con_name: str, target: str | None
) -> None:
    """Only a plain local path is ours to check or to open with a mode. duckdb
    refuses `:memory:` read-only outright, a MotherDuck handle is no path, and a
    recorded URI already spells its own mode."""
    profile = Profile(con_name=con_name, kwargs_tuple=(("database", target),))

    assert missing_database_file(profile) is None
    assert no_create_kwargs(profile) == {}
