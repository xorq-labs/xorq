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

from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
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
