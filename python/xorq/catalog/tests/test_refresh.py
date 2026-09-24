"""``catalog.refresh``: re-derive a loaded build over its drifted sources (#2320).

The cases are #2327's, ported from ``load_expr(..., refresh_schemas=True)`` to
``refresh_build``, plus the ones only a post-load rewrite can get wrong. sqlite
for the reason ``test_drift.py`` uses it: it survives a build as a real
``DatabaseTable`` with a live source behind it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog.drift import iter_leaf_reports
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import (
    live_schemas,
    refresh_build,
    refresh_schemas,
    with_live_schema,
)
from xorq.common.exceptions import SchemaRefreshError
from xorq.common.utils.defer_utils import deferred_read_csv, deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import CachedNode, Read, Tag, TeeNode, pin_cache
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.ibis_yaml.enums import ReadKwarg
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.writes import ParquetWriteThrough


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})
GROWN = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))


def recreate(con: SqliteBackend, table: pa.Table, name: str = "t") -> None:
    """Put a differently shaped table where the build's was. sqlite has no
    ALTER COLUMN TYPE, so a retype is a drop and recreate like everything else."""
    con.drop_table(name, force=True)
    con.create_table(name, table.to_pandas())


@pytest.fixture
def builds_dir(tmp_path: Path) -> Path:
    return tmp_path / "builds"


@pytest.fixture
def con(tmp_path: Path) -> SqliteBackend:
    con = SqliteBackend().connect(str(tmp_path / "live.sqlite"))
    con.create_table("t", RECORDED.to_pandas())
    return con


@pytest.fixture
def world(con: SqliteBackend, builds_dir: Path) -> tuple:
    """A build of `t.filter(t.a > 1)`: `a` is referenced, `b` is not."""
    t = con.table("t")
    return con, build_expr(t.filter(t.a > 1), builds_dir=builds_dir)


def write_parquet(path: Path, table: pa.Table) -> None:
    table.to_pandas().to_parquet(path, index=False)


def test_an_unchanged_world_refreshes_to_the_recorded_schema(world: tuple) -> None:
    _, build_path = world
    assert refresh_build(build_path).schema() == load_expr(build_path).schema()


def test_an_added_column_reaches_the_expression(world: tuple) -> None:
    con, build_path = world
    recreate(con, GROWN)

    assert "c" not in load_expr(build_path).schema()
    assert str(refresh_build(build_path).schema()["c"]) == "float64"


def test_a_retyped_column_reaches_the_expression(world: tuple) -> None:
    con, build_path = world
    recreate(con, pa.table({"a": pa.array([1.0, 2.0], pa.float64()), "b": ["x", "y"]}))

    assert str(load_expr(build_path).schema()["a"]) == "int64"
    assert str(refresh_build(build_path).schema()["a"]) == "float64"


def test_an_unreferenced_column_dropped_still_rebuilds(world: tuple) -> None:
    con, build_path = world
    recreate(con, RECORDED.drop_columns("b"))

    schema = refresh_build(build_path).schema()
    assert "b" not in schema
    assert "a" in schema


def test_a_dropped_referenced_column_names_the_field(world: tuple) -> None:
    con, build_path = world
    recreate(con, RECORDED.drop_columns("a"))

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    # The deepest op that could not be reconstructed, not the `Filter` above it.
    assert excinfo.value.op_name == "Field"
    assert excinfo.value.cause is not None


def test_a_numeric_column_turned_string_fails_its_aggregate(
    con: SqliteBackend, builds_dir: Path
) -> None:
    """`Mean` takes a numeric column; the rebuild re-runs that signature."""
    t = con.table("t")
    build_path = build_expr(t.group_by("b").agg(m=t.a.mean()), builds_dir=builds_dir)
    recreate(con, pa.table({"a": ["1", "x"], "b": ["x", "y"]}))

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == "Mean"


def test_the_recorded_path_survives_a_world_it_cannot_load(world: tuple) -> None:
    """Refreshing is a separate step: the build still loads as recorded."""
    con, build_path = world
    recreate(con, RECORDED.drop_columns("a"))

    assert dict(load_expr(build_path).schema()) == dict(
        xo.schema({"a": "int64", "b": "string"})
    )


def test_only_the_drifted_source_moves(con: SqliteBackend, builds_dir: Path) -> None:
    """The undrifted side of a join rebuilds to the node it was."""
    con.create_table("u", RECORDED.to_pandas())
    t, u = con.table("t"), con.table("u")
    renamed = u.select(k=u.a, v=u.b)
    build_path = build_expr(t.join(renamed, t.a == renamed.k), builds_dir=builds_dir)
    recreate(con, GROWN)

    record = BuildRecord.from_build_dir(build_path)
    live = live_schemas(record, iter_leaf_reports(record))
    loaded = load_expr(build_path)
    refreshed = refresh_schemas(loaded, live)
    (before_u,) = (n for n in walk_nodes(ops.DatabaseTable, loaded) if n.name == "u")
    (after_u,) = (n for n in walk_nodes(ops.DatabaseTable, refreshed) if n.name == "u")
    (after_t,) = (n for n in walk_nodes(ops.DatabaseTable, refreshed) if n.name == "t")
    assert after_u is before_u
    assert "c" in after_t.schema
    # A join records its output columns, so an added one stops at the source.
    assert refreshed.schema() == loaded.schema()


def test_a_missing_source_stops_before_loading(world: tuple) -> None:
    con, build_path = world
    con.drop_table("t")

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == "DatabaseTable"


def test_a_deleted_database_is_not_recreated(world: tuple, tmp_path: Path) -> None:
    """The sweep's guarded connection is what finds it gone."""
    _, build_path = world
    db = tmp_path / "live.sqlite"
    db.unlink()

    with pytest.raises(SchemaRefreshError):
        refresh_build(build_path)
    assert not db.exists()


def test_a_refreshed_read_is_still_a_read(tmp_path: Path, builds_dir: Path) -> None:
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    build_path = build_expr(
        deferred_read_parquet(path, xo.connect(), table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    write_parquet(path, GROWN)

    expr = refresh_build(build_path)
    (read,) = walk_nodes(Read, expr)
    assert read.method_name == "read_parquet"
    assert "c" in read.schema
    assert str(expr.schema()["c"]) == "float64"


def test_a_bundled_read_keeps_its_recorded_schema(
    tmp_path: Path, builds_dir: Path
) -> None:
    """Its bytes are in the archive, so it is drift-exempt and never probed."""
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    build_path = build_expr(
        deferred_read_parquet(path, xo.connect(), table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=True,
    )
    path.unlink()

    assert refresh_build(build_path).schema() == load_expr(build_path).schema()


def test_a_cached_node_follows_its_parent(
    con: SqliteBackend, tmp_path: Path, builds_dir: Path
) -> None:
    t = con.table("t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    build_path = build_expr(t.filter(t.a > 1).cache(cache=cache), builds_dir=builds_dir)
    recreate(con, GROWN)

    (cached,) = walk_nodes(CachedNode, refresh_build(build_path))
    assert dict(cached.schema) == dict(con.table("t").schema())


def test_a_tee_node_follows_its_parent(tmp_path: Path, builds_dir: Path) -> None:
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    expr = deferred_read_parquet(path, xo.connect(), table_name="t").tee(
        ParquetWriteThrough(path=tmp_path / "out.parquet")
    )
    build_path = build_expr(expr, builds_dir=builds_dir, relocate_reads=False)
    write_parquet(path, GROWN)

    (tee,) = walk_nodes(TeeNode, refresh_build(build_path))
    assert tuple(tee.schema) == tuple(GROWN.schema.names)


def test_a_tagged_build_refreshes_through_the_tag(
    con: SqliteBackend, builds_dir: Path
) -> None:
    tagged = con.table("t").tag("step")
    build_path = build_expr(tagged.filter(tagged.a > 1), builds_dir=builds_dir)
    recreate(con, GROWN)

    expr = refresh_build(build_path)
    (tag,) = walk_nodes(Tag, expr)
    assert "c" in expr.schema()
    assert tag.schema == tag.parent.schema


def test_a_tag_does_not_mask_a_dropped_column(
    con: SqliteBackend, builds_dir: Path
) -> None:
    tagged = con.table("t").tag("step")
    build_path = build_expr(tagged.filter(tagged.a > 1), builds_dir=builds_dir)
    recreate(con, RECORDED.drop_columns("a"))

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == "Field"


def test_a_csv_read_refreshes(tmp_path: Path, builds_dir: Path) -> None:
    """The recorded schema is also a `read_kwargs` instruction, and is rewritten
    so the node does not advertise columns it would then decline to read."""
    path = tmp_path / "t.csv"
    RECORDED.to_pandas().to_csv(path, index=False)
    build_path = build_expr(
        deferred_read_csv(path, xo.connect(), table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    GROWN.to_pandas().to_csv(path, index=False)

    expr = refresh_build(build_path)
    (read,) = walk_nodes(Read, expr)
    assert "c" in expr.schema()
    assert "c" in dict(read.read_kwargs)[ReadKwarg.schema]


def test_refresh_replaces_a_declared_schema_with_inference(
    tmp_path: Path, builds_dir: Path
) -> None:
    """The known limitation, shared with `catalog.drift`: a declared schema is
    recorded like an inferred one, so it reads as drift and is replaced."""
    path = tmp_path / "t.csv"
    RECORDED.to_pandas().to_csv(path, index=False)
    declared = xo.schema({"a": "string", "b": "string"})
    build_path = build_expr(
        deferred_read_csv(path, xo.connect(), table_name="t", schema=declared),
        builds_dir=builds_dir,
        relocate_reads=False,
    )

    assert str(load_expr(build_path).schema()["a"]) == "string"
    expr = refresh_build(build_path)
    (read,) = walk_nodes(Read, expr)
    assert str(expr.schema()["a"]) == "int64"
    assert str(dict(read.read_kwargs)[ReadKwarg.schema]["a"]) == "int64"


def test_a_refresh_drops_a_stale_types_override(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    RECORDED.to_pandas().to_csv(path, index=False)
    (read,) = walk_nodes(
        Read,
        deferred_read_csv(
            path,
            xo.duckdb.connect(),
            table_name="t",
            schema=xo.schema({"a": "string", "b": "string"}),
            types=FrozenDict({"a": "VARCHAR"}),
        ),
    )

    kwargs = dict(with_live_schema(read, xo.schema(RECORDED.schema)).read_kwargs)
    assert ReadKwarg.types not in kwargs
    assert str(kwargs[ReadKwarg.columns]["a"]) == "int64"


def test_a_refresh_does_not_write_to_the_source(
    tmp_path: Path, builds_dir: Path
) -> None:
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    db = str(tmp_path / "live.sqlite")
    con = SqliteBackend().connect(db)
    build_path = build_expr(
        deferred_read_parquet(path, con, table_name="ingested"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    con.create_table("ingested", pa.table({"precious": [42]}).to_pandas())
    write_parquet(path, GROWN)

    assert "c" in refresh_build(build_path).schema()
    after = SqliteBackend().connect(db)
    assert list(after.table("ingested").schema()) == ["precious"]


def test_a_pinned_build_stays_self_contained(tmp_path: Path, builds_dir: Path) -> None:
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    t = deferred_read_parquet(path, xo.connect(), table_name="t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    build_path = build_expr(
        pin_cache(t.filter(t.a > 1).cache(cache=cache), ensure_materialized=True),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    path.unlink()

    assert refresh_build(build_path).schema() == load_expr(build_path).schema()


def test_an_empty_mapping_is_the_identity(world: tuple) -> None:
    _, build_path = world
    expr = load_expr(build_path)
    assert refresh_schemas(expr, {}) is expr


def test_a_refresh_error_survives_a_process_boundary() -> None:
    err = pickle.loads(pickle.dumps(SchemaRefreshError("Field", ValueError("boom"))))
    assert err.op_name == "Field"
    assert isinstance(err.cause, ValueError)
    assert "Field" in str(err)
