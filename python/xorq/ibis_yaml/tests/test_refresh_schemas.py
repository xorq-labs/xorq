"""``load_expr(..., refresh_schemas=True)`` (xorq-labs/xorq#2320).

sqlite for the same reason `catalog/tests/test_drift.py` uses it: it survives a
build as a real ``DatabaseTable`` with a live source behind it, where duckdb is
a memory backend whose tables are materialized into the archive and have
nothing to drift against.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.common.exceptions import SchemaRefreshError
from xorq.common.utils.defer_utils import deferred_read_csv, deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import CachedNode, Read, Tag, TeeNode, pin_cache
from xorq.ibis_yaml.common import TranslationContext
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.ibis_yaml.enums import ReadKwarg
from xorq.ibis_yaml.utils import namespace_to_database
from xorq.writes import ParquetWriteThrough


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


def recreate(con, table: pa.Table, name: str = "t") -> None:
    """Put a differently shaped `t` where the build's `t` was. sqlite has no
    ALTER COLUMN TYPE, so a retype is a drop and recreate like everything else."""
    con.drop_table(name, force=True)
    con.create_table(name, table.to_pandas())


@pytest.fixture
def world(tmp_path: Path, builds_dir: Path):
    """A sqlite table `t`, and a build of `t.filter(t.a > 1)` over it.

    The filter's only `Field` is `a`, so `a` is the referenced column and `b`
    the unreferenced one.
    """
    con = SqliteBackend().connect(str(tmp_path / "live.sqlite"))
    con.create_table("t", RECORDED.to_pandas())
    t = con.table("t")
    return con, build_expr(t.filter(t.a > 1), builds_dir=builds_dir)


def test_an_unchanged_world_refreshes_to_the_recorded_schema(world) -> None:
    _, build_path = world
    assert load_expr(build_path, refresh_schemas=True).schema() == (
        load_expr(build_path).schema()
    )


def test_an_added_column_reaches_the_expression(world) -> None:
    con, build_path = world
    recreate(con, RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64())))

    assert "c" not in load_expr(build_path).schema()
    assert str(load_expr(build_path, refresh_schemas=True).schema()["c"]) == "float64"


def test_a_retyped_column_reaches_the_expression(world) -> None:
    con, build_path = world
    recreate(con, pa.table({"a": pa.array([1.0, 2.0], pa.float64()), "b": ["x", "y"]}))

    assert str(load_expr(build_path).schema()["a"]) == "int64"
    assert str(load_expr(build_path, refresh_schemas=True).schema()["a"]) == "float64"


def test_an_unreferenced_column_dropped_still_rebuilds(world) -> None:
    con, build_path = world
    recreate(con, RECORDED.drop_columns("b"))

    schema = load_expr(build_path, refresh_schemas=True).schema()
    assert "b" not in schema
    assert "a" in schema


def test_a_dropped_referenced_column_names_the_field(world) -> None:
    con, build_path = world
    recreate(con, RECORDED.drop_columns("a"))

    with pytest.raises(SchemaRefreshError) as excinfo:
        load_expr(build_path, refresh_schemas=True)
    # The deepest op that could not be reconstructed, not the `Filter` above it.
    assert excinfo.value.op_name == "Field"
    assert "Field" in str(excinfo.value)
    assert excinfo.value.cause is not None


def test_the_recorded_path_survives_a_world_it_cannot_load(world) -> None:
    """The mode is opt-in: the same build still loads against the old schema."""
    con, build_path = world
    recreate(con, RECORDED.drop_columns("a"))

    assert dict(load_expr(build_path).schema()) == dict(
        xo.schema({"a": "int64", "b": "string"})
    )


def test_a_refreshed_read_is_still_a_read(tmp_path: Path, builds_dir: Path) -> None:
    """The read is recreated with the live schema rather than replaced by a
    `make_dt()` `DatabaseTable`, which would drop the method and kwargs out of
    the rebuilt record."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    con = xo.connect()
    build_path = build_expr(
        deferred_read_parquet(path, con, table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )

    grown = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))
    grown.to_pandas().to_parquet(path, index=False)

    expr = load_expr(build_path, refresh_schemas=True)
    (read,) = walk_nodes(Read, expr)
    assert read.method_name == "read_parquet"
    assert "c" in read.schema
    assert str(expr.schema()["c"]) == "float64"


def test_a_bundled_read_keeps_its_recorded_schema(
    tmp_path: Path, builds_dir: Path
) -> None:
    """Its bytes are in the archive, so it cannot drift and the mode must not
    resolve its build-relative path against the process working directory.
    Refreshing it at all is xorq-labs/xorq#2322."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    con = xo.connect()
    build_path = build_expr(
        deferred_read_parquet(path, con, table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=True,
    )
    path.unlink()

    assert load_expr(build_path, refresh_schemas=True).schema() == (
        load_expr(build_path).schema()
    )


def test_a_cached_node_follows_its_parent(tmp_path: Path, builds_dir: Path) -> None:
    """Nothing validates a CachedNode against its parent, so a stale schema
    survives a load and the node advertises columns it would not produce."""
    con = SqliteBackend().connect(str(tmp_path / "live.sqlite"))
    con.create_table("t", RECORDED.to_pandas())
    t = con.table("t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    build_path = build_expr(t.filter(t.a > 1).cache(cache=cache), builds_dir=builds_dir)

    recreate(con, RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64())))

    expr = load_expr(build_path, refresh_schemas=True)
    (cached,) = walk_nodes(CachedNode, expr)
    # Against the live source, not against the parent the handler just read the
    # schema off -- that comparison cannot fail.
    assert dict(cached.schema) == dict(con.table("t").schema())


def test_a_tee_node_follows_its_parent(tmp_path: Path, builds_dir: Path) -> None:
    """`TeeNode.__init__` rejects a schema that is not its parent's, so without
    the refresh branch an additive change fails as an integrity error rather
    than rebuilding."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    con = xo.connect()
    expr = deferred_read_parquet(path, con, table_name="t").tee(
        ParquetWriteThrough(path=tmp_path / "out.parquet")
    )
    build_path = build_expr(expr, builds_dir=builds_dir, relocate_reads=False)

    grown = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))
    grown.to_pandas().to_parquet(path, index=False)

    loaded = load_expr(build_path, refresh_schemas=True)
    (tee,) = walk_nodes(TeeNode, loaded)
    # Against the grown file, not against `tee.parent` -- `TeeNode.__init__`
    # would have raised an `IntegrityError` before any such assert could run.
    assert tuple(tee.schema) == tuple(grown.schema.names)


def test_a_missing_source_is_not_swallowed(world: tuple) -> None:
    """The mode dials, so a source that is gone surfaces as a refresh failure
    naming the op that needed it rather than a silently recorded schema."""
    con, build_path = world
    con.drop_table("t")

    with pytest.raises(SchemaRefreshError) as excinfo:
        load_expr(build_path, refresh_schemas=True)
    assert excinfo.value.op_name == "DatabaseTable"


def test_a_tagged_build_refreshes_through_the_tag(
    tmp_path: Path, builds_dir: Path
) -> None:
    """`Tag` is transparent like `TeeNode` with nothing enforcing it, so a
    recorded schema over a refreshed parent would stop the refresh dead --
    every catalog bind and every ML pipeline step carries one."""
    con = SqliteBackend().connect(str(tmp_path / "live.sqlite"))
    con.create_table("t", RECORDED.to_pandas())
    t = con.table("t")
    tagged = t.tag("step")
    build_path = build_expr(tagged.filter(tagged.a > 1), builds_dir=builds_dir)

    recreate(con, RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64())))

    expr = load_expr(build_path, refresh_schemas=True)
    (tag,) = walk_nodes(Tag, expr)
    assert "c" in expr.schema()
    assert tag.schema == tag.parent.schema


def test_a_tag_does_not_mask_a_dropped_column(tmp_path: Path, builds_dir: Path) -> None:
    """The failure the mode exists to raise must not be absorbed by a stale
    schema one node above the source."""
    con = SqliteBackend().connect(str(tmp_path / "live.sqlite"))
    con.create_table("t", RECORDED.to_pandas())
    t = con.table("t")
    tagged = t.tag("step")
    build_path = build_expr(tagged.filter(tagged.a > 1), builds_dir=builds_dir)

    recreate(con, RECORDED.drop_columns("a"))

    with pytest.raises(SchemaRefreshError) as excinfo:
        load_expr(build_path, refresh_schemas=True)
    assert excinfo.value.op_name == "Field"


def test_a_csv_read_refreshes(tmp_path: Path, builds_dir: Path) -> None:
    """`deferred_read_csv` records its inferred schema as a `read_kwargs`
    *instruction* the read method obeys, so replaying the read hands the
    recorded answer back. The refresh re-runs the build-time inference instead,
    and rewrites the instruction so the node does not advertise columns it
    would then decline to read."""
    path = tmp_path / "t.csv"
    RECORDED.to_pandas().to_csv(path, index=False)
    con = xo.connect()
    build_path = build_expr(
        deferred_read_csv(path, con, table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )

    grown = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))
    grown.to_pandas().to_csv(path, index=False)

    expr = load_expr(build_path, refresh_schemas=True)
    (read,) = walk_nodes(Read, expr)
    assert "c" in expr.schema()
    assert "c" in dict(read.read_kwargs)[ReadKwarg.schema]


def test_a_refresh_does_not_write_to_the_source(
    tmp_path: Path, builds_dir: Path
) -> None:
    """Replaying a read keeps the recorded `table_name` and the `mode="replace"`
    the ADBC backends record, so a load would drop and re-ingest a table in the
    user's database. The refresh reads the file, never the connection."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    db = str(tmp_path / "live.sqlite")
    con = SqliteBackend().connect(db)
    build_path = build_expr(
        deferred_read_parquet(path, con, table_name="ingested"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    con.create_table("ingested", pa.table({"precious": [42]}).to_pandas())

    grown = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))
    grown.to_pandas().to_parquet(path, index=False)

    assert "c" in load_expr(build_path, refresh_schemas=True).schema()
    after = SqliteBackend().connect(db)
    assert list(after.table("ingested").schema()) == ["precious"]


def test_a_pinned_build_stays_self_contained(tmp_path: Path, builds_dir: Path) -> None:
    """A pin is drift-exempt and documented to load without its original
    sources, so the refresh must not dial the upstream `CacheTag` discarded."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    con = xo.connect()
    t = deferred_read_parquet(path, con, table_name="t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    build_path = build_expr(
        pin_cache(t.filter(t.a > 1).cache(cache=cache), ensure_materialized=True),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    path.unlink()

    assert load_expr(build_path, refresh_schemas=True).schema() == (
        load_expr(build_path).schema()
    )


def test_a_refresh_error_survives_a_process_boundary() -> None:
    """`BaseException.__reduce__` rebuilds from `args`, so a pre-formatted
    single arg would make the error unpicklable and a refresh failure crossing
    a Flight or subprocess boundary would surface as the unpickler's
    `TypeError`."""
    err = pickle.loads(pickle.dumps(SchemaRefreshError("Field", ValueError("boom"))))
    assert err.op_name == "Field"
    assert isinstance(err.cause, ValueError)
    assert "Field" in str(err)


def test_the_context_field_reaches_the_cache_key() -> None:
    """`translate_from_yaml` is lru_cached on (yaml_dict, context); a context
    that excluded the flag from its identity would let a refreshed load collect
    another load's recorded-schema expression."""
    registry_shared = TranslationContext().registry
    off = TranslationContext(registry=registry_shared)
    on = TranslationContext(registry=registry_shared, refresh_schemas=True)
    assert off != on
    assert hash(off) != hash(on)


@pytest.mark.parametrize(
    "catalog,database,expected",
    [
        (None, None, None),
        ("", "", None),
        (None, "main", "main"),
        ("cat", "db", ("cat", "db")),
    ],
)
def test_a_namespace_maps_the_way_the_probe_maps_it(catalog, database, expected):
    """One mapping, shared with `drift.table_location`, so what the probe asks a
    backend for and what a refresh rebuilds over are the same place."""
    assert namespace_to_database(catalog, database) == expected


def test_a_catalog_without_a_database_raises():
    with pytest.raises(ValueError, match="without a database"):
        namespace_to_database("cat", None)
