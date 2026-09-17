"""`check-sources` over `Read` leaves (xorq-labs/xorq#2296).

`relocate_reads` defaults to True and relocation copies the bytes into the
archive, so the *default* build of a local read is bundled and exempt. What is
left to check is a read built with relocation off, and a read whose path is a
remote URI, which relocation refuses and leaves untouched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import checkable_leaves, make_profile, read_call
from xorq.catalog.inspection import BuildRecord, iter_source_leaves


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


def read_doc(hash_path: str, extra: tuple = ()) -> dict:
    """A one-`Read` document, for the shapes a build cannot easily produce."""
    return {
        "definitions": {
            "dtypes": {},
            "nodes": {
                "@read_0": {
                    "op": "Read",
                    "name": "src",
                    "method_name": "read_parquet",
                    "profile": "p0",
                    "read_kwargs": [
                        ["hash_path", hash_path],
                        ["table_name", "src"],
                        *extra,
                    ],
                    "schema_ref": "schema_0",
                }
            },
            "schemas": {
                "schema_0": {"a": {"op": "DataType", "type": "Int64", "nullable": True}}
            },
        },
        "expression": {"node_ref": "@read_0"},
    }


@pytest.fixture
def world(tmp_path: Path, catalog_path: str) -> SimpleNamespace:
    """One parquet file, built into a catalog twice: relocation off, then on."""
    src = tmp_path / "src.parquet"
    pq.write_table(RECORDED, src)
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    read = xo.deferred_read_parquet(src, xo.connect(), table_name="src")
    expr = read.filter(read.a > 1)
    return SimpleNamespace(
        src=src,
        catalog=catalog,
        catalog_path=catalog_path,
        external=catalog.add(expr, relocate_reads=False).name,
        bundled=catalog.add(expr).name,
    )


def check_sources(runner: CliRunner, world: SimpleNamespace, name: str):
    return runner.invoke(cli, ["--path", world.catalog_path, "check-sources", name])


def test_an_unrelocated_read_is_checked_by_path(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    result = check_sources(runner, world, world.external)
    assert result.exit_code == 0
    assert f"Read {world.src}: equal" in result.output


def test_a_moved_file_is_missing(runner: CliRunner, world: SimpleNamespace) -> None:
    world.src.rename(world.src.with_name("moved.parquet"))

    result = check_sources(runner, world, world.external)
    assert result.exit_code == 3
    assert f"Read {world.src}: table-missing" in result.output
    assert "live:     -" in result.output


def test_an_added_column_is_changed(runner: CliRunner, world: SimpleNamespace) -> None:
    pq.write_table(RECORDED.append_column("c", pa.array([1.0, 2.0])), world.src)

    result = check_sources(runner, world, world.external)
    assert result.exit_code == 3
    assert "changed" in result.output
    assert "live:     a int64, b string, c float64" in result.output


def test_a_relocated_read_is_exempt(runner: CliRunner, world: SimpleNamespace) -> None:
    """Its bytes are in the archive, so the source file no longer speaks for it."""
    pq.write_table(RECORDED.append_column("c", pa.array([1.0, 2.0])), world.src)

    result = check_sources(runner, world, world.bundled)
    assert result.exit_code == 0
    assert "no external sources (1 reads)" in result.output


def test_a_remote_uri_is_checkable() -> None:
    """Relocation refuses remote schemes, so the real URI survives the build."""
    record = BuildRecord(read_doc("s3://bucket/src.parquet"), {})
    (leaf,) = checkable_leaves(record)
    assert leaf.name == "s3://bucket/src.parquet"


def test_the_relocation_keys_are_not_replayed() -> None:
    """`READ_EXCLUDE_KEYS`, not a second spelling of the same list."""
    (leaf,) = iter_source_leaves(
        read_doc(
            "/data/src.parquet",
            extra=(["read_path", "reads/abc.parquet"], ["relocatable", True]),
        )
    )
    assert read_call(leaf) == (("/data/src.parquet",), {"table_name": "src"})


def test_probing_leaves_nothing_behind(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The read routes to a session-scoped view that dies with the connection."""
    assert check_sources(runner, world, world.external).exit_code == 0

    record = BuildRecord.from_catalog_entry(
        world.catalog.get_catalog_entry(world.external)
    )
    (leaf,) = record.external_leaves
    con = make_profile(record.get_profile_dict(leaf)).get_con()
    assert "src" not in con.list_tables()
