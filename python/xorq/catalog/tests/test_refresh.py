"""``catalog.refresh`` (#2320). sqlite: it builds to a real ``DatabaseTable``."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import pyarrow as pa
import pytest
import toolz

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog.drift import (
    LeafReport,
    close_cons,
    iter_leaf_reports,
    unchecked_leaves,
)
from xorq.catalog.enums import LeafKind, Verdict
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import (
    check_refreshable,
    leaf_key,
    live_schemas,
    op_key,
    refresh_build,
    refresh_schemas,
    with_live_schema,
)
from xorq.common.exceptions import InternalError, RefreshCause, SchemaRefreshError
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
    make_read_kwargs,
    normalize_read_path_stat,
)
from xorq.common.utils.graph_utils import walk_nodes
from xorq.common.utils.node_utils import recreate
from xorq.common.utils.provenance_utils import get_expr_hash
from xorq.expr.relations import (
    CachedNode,
    FlightExpr,
    FlightUDXF,
    Read,
    RemoteTable,
    Tag,
    TeeNode,
    flight_expr,
    flight_udxf,
    pin_cache,
)
from xorq.flight import FlightServer
from xorq.flight.tests.test_server import make_flight_url
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.ibis_yaml.enums import ReadKwarg
from xorq.vendor.ibis.common.annotations import ValidationError
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.expr.types import Expr
from xorq.writes import ParquetWriteThrough


if TYPE_CHECKING:
    from xorq.backends.duckdb import Backend as DuckDBBackend


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})
GROWN = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))


RETYPED = RECORDED.set_column(0, "a", pa.array([1.0, 2.0], pa.float64()))
STRINGY = pa.table({"a": ["1", "x"], "b": ["x", "y"]})


def replace_table(con: SqliteBackend, table: pa.Table, name: str = "t") -> None:
    """Replace table ``name`` with ``table`` (sqlite has no ALTER COLUMN TYPE)."""
    con.drop_table(name, force=True)
    con.create_table(name, table.to_pandas())


@pytest.fixture
def builds_dir(tmp_path: Path) -> Path:
    return tmp_path / "builds"


@pytest.fixture
def con(tmp_path: Path) -> Iterator[SqliteBackend]:
    con = SqliteBackend().connect(str(tmp_path / "live.sqlite"))
    con.create_table("t", RECORDED.to_pandas())
    yield con
    con.disconnect()


@pytest.fixture
def world(con: SqliteBackend, builds_dir: Path) -> tuple:
    """A build of `t.filter(t.a > 1)`: `a` is referenced, `b` is not."""
    t = con.table("t")
    return con, build_expr(t.filter(t.a > 1), builds_dir=builds_dir)


@pytest.fixture
def world_leaf(world: tuple) -> tuple:
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    return build_path, record, leaf


def write_parquet(path: Path, table: pa.Table) -> None:
    table.to_pandas().to_parquet(path, index=False)


def read_json(con: DuckDBBackend, path: Path, name: str, **kwargs: Any) -> ir.Table:
    """A duckdb ``read_json`` recorded at ``a: int64``; it has no inference."""
    return Read(
        method_name="read_json",
        name=name,
        schema=xo.schema({"a": "int64"}),
        source=con,
        read_kwargs=make_read_kwargs(
            con.read_json, str(path), table_name=name, **kwargs
        ),
        normalize_method=normalize_read_path_stat,
    ).to_expr()


def test_an_unchanged_world_refreshes_to_the_recorded_schema(world: tuple) -> None:
    _, build_path = world
    assert refresh_build(build_path).schema() == load_expr(build_path).schema()


@pytest.mark.parametrize(
    ("table", "column"),
    (
        pytest.param(GROWN, "c", id="added"),
        pytest.param(RETYPED, "a", id="retyped"),
    ),
)
def test_a_drifted_column_reaches_the_expression(
    world: tuple, table: pa.Table, column: str
) -> None:
    con, build_path = world
    replace_table(con, table)
    assert str(refresh_build(build_path).schema()[column]) == "float64"


def test_an_unreferenced_column_dropped_still_rebuilds(world: tuple) -> None:
    con, build_path = world
    replace_table(con, RECORDED.drop_columns("b"))
    assert list(refresh_build(build_path).schema()) == ["a"]


def filter_through_a_tag(t: ir.Table) -> ir.Table:
    tagged = t.tag("step")
    return tagged.filter(tagged.a > 1)


# The deepest op that no longer fits is named, not the `Filter` above it.
@pytest.mark.parametrize(
    ("make_expr", "table", "op_name"),
    (
        pytest.param(
            lambda t: t.filter(t.a > 1), RECORDED.drop_columns("a"), "Field", id="field"
        ),
        pytest.param(
            filter_through_a_tag, RECORDED.drop_columns("a"), "Field", id="tag"
        ),
        # `DropColumns` checks its columns in its `schema` attribute.
        pytest.param(
            lambda t: t.drop("b"),
            RECORDED.drop_columns("b"),
            "DropColumns",
            id="drop-columns",
        ),
        pytest.param(
            lambda t: t.group_by("b").agg(m=t.a.mean()), STRINGY, "Mean", id="mean"
        ),
        # `SimpleCase` validates in its constructor, not its signature.
        pytest.param(
            lambda t: t.mutate(c=t.a.cases((1, "one"), else_="other")),
            STRINGY,
            "SimpleCase",
            id="simple-case",
        ),
    ),
)
def test_an_op_that_no_longer_fits_is_named(
    con: SqliteBackend,
    builds_dir: Path,
    make_expr: Callable,
    table: pa.Table,
    op_name: str,
) -> None:
    build_path = build_expr(make_expr(con.table("t")), builds_dir=builds_dir)
    replace_table(con, table)

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == op_name


def test_the_recorded_path_survives_a_world_it_cannot_load(world: tuple) -> None:
    """Refreshing is a separate step: the build still loads as recorded."""
    con, build_path = world
    replace_table(con, RECORDED.drop_columns("a"))

    assert dict(load_expr(build_path).schema()) == dict(
        xo.schema({"a": "int64", "b": "string"})
    )


def test_only_the_drifted_source_moves(con: SqliteBackend, builds_dir: Path) -> None:
    """The undrifted side of a join rebuilds to the node it was."""
    con.create_table("u", RECORDED.to_pandas())
    t, u = con.table("t"), con.table("u")
    renamed = u.select(k=u.a, v=u.b)
    build_path = build_expr(t.join(renamed, t.a == renamed.k), builds_dir=builds_dir)
    replace_table(con, GROWN)

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


def test_one_table_name_on_two_connections_is_two_sources(
    con: SqliteBackend, tmp_path: Path, builds_dir: Path
) -> None:
    other = SqliteBackend().connect(str(tmp_path / "other.sqlite"))
    other.create_table("t", RECORDED.to_pandas())
    t, u = con.table("t"), other.table("t")
    remote = u.select(k=u.a, v=u.b).into_backend(con)
    build_path = build_expr(t.join(remote, t.a == remote.k), builds_dir=builds_dir)
    replace_table(con, GROWN)

    schemas = {
        node.source._profile.kwargs_dict["database"]: node.schema
        for node in walk_nodes(ops.DatabaseTable, refresh_build(build_path))
        if type(node) is ops.DatabaseTable
    }
    assert "c" in schemas[str(tmp_path / "live.sqlite")]
    assert "c" not in schemas[str(tmp_path / "other.sqlite")]


def test_a_lazy_load_connects_only_the_drifted_source(
    con: SqliteBackend, tmp_path: Path, builds_dir: Path
) -> None:
    """An undrifted source is never connected."""
    other_path = tmp_path / "other.sqlite"
    other = SqliteBackend().connect(str(other_path))
    other.create_table("u", RECORDED.to_pandas())
    t, u = con.table("t"), other.table("u")
    remote = u.select(k=u.a, v=u.b).into_backend(con)
    build_path = build_expr(t.join(remote, t.a == remote.k), builds_dir=builds_dir)
    replace_table(con, GROWN)
    record = BuildRecord.from_build_dir(build_path)
    live = live_schemas(record, iter_leaf_reports(record))
    # A directory where the database was: connecting to it raises.
    other.disconnect()
    other_path.unlink()
    other_path.mkdir()

    refreshed = refresh_schemas(load_expr(build_path, lazy=True), live)
    (after_t,) = (n for n in walk_nodes(ops.DatabaseTable, refreshed) if n.name == "t")
    assert "c" in after_t.schema


def test_every_missing_source_is_named(con: SqliteBackend, builds_dir: Path) -> None:
    con.create_table("u", RECORDED.to_pandas())
    t, u = con.table("t"), con.table("u")
    build_path = build_expr(t.union(u), builds_dir=builds_dir)
    con.drop_table("t")
    con.drop_table("u")

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == "DatabaseTable"
    assert "t is table-missing" in str(excinfo.value)
    assert "u is table-missing" in str(excinfo.value)


def test_a_deleted_database_is_not_recreated(world: tuple, tmp_path: Path) -> None:
    """The sweep's guarded connection is what finds it gone."""
    _, build_path = world
    db = tmp_path / "live.sqlite"
    db.unlink()

    with pytest.raises(SchemaRefreshError):
        refresh_build(build_path)
    assert not db.exists()


@pytest.mark.parametrize("error", (AttributeError, InternalError))
def test_a_bug_in_the_rewrite_is_not_labeled_as_drift(
    world: tuple, monkeypatch: pytest.MonkeyPatch, error: type
) -> None:
    con, build_path = world
    replace_table(con, GROWN)

    def broken(node, **kwargs):
        raise error("a bug in the rewrite")

    monkeypatch.setattr("xorq.catalog.refresh.recreate", broken)
    with pytest.raises(error, match="a bug in the rewrite") as excinfo:
        refresh_build(build_path)
    assert not isinstance(excinfo.value, SchemaRefreshError)


def test_a_refreshed_build_hashes_like_a_fresh_build(world: tuple) -> None:
    """What a rebase's no-op check compares: the refresh is the live build."""
    con, build_path = world
    replace_table(con, GROWN)

    t = con.table("t")
    assert get_expr_hash(refresh_build(build_path)) == get_expr_hash(t.filter(t.a > 1))


def test_a_namespaced_table_keys_like_its_leaf(
    con: SqliteBackend, builds_dir: Path
) -> None:
    """The recorded and loaded spellings of a dotted name must agree."""
    table = recreate(con.table("t").op(), namespace=ops.Namespace(database="s"))
    build_path = build_expr(table.to_expr(), builds_dir=builds_dir)
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    (loaded,) = (
        node
        for node in walk_nodes(ops.DatabaseTable, load_expr(build_path))
        if type(node) is ops.DatabaseTable
    )

    assert leaf.name == "s.t"
    assert op_key(loaded) == leaf_key(leaf, record)


def test_a_caller_owned_con_cache_is_left_open(world: tuple) -> None:
    con, build_path = world
    replace_table(con, GROWN)
    con_cache: dict = {}
    try:
        assert "c" in refresh_build(build_path, con_cache=con_cache).schema()
        (sweep_con,) = con_cache.values()
        assert sweep_con.list_tables() == ["t"]
    finally:
        close_cons(con_cache)


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


def test_a_multi_path_read_refreshes(tmp_path: Path, builds_dir: Path) -> None:
    paths = (tmp_path / "1.parquet", tmp_path / "2.parquet")
    for path in paths:
        write_parquet(path, RECORDED)
    build_path = build_expr(
        deferred_read_parquet(tuple(map(str, paths)), xo.connect(), table_name="t"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    for path in paths:
        write_parquet(path, GROWN)

    assert list(refresh_build(build_path).execute()["c"]) == [1.5, 2.5, 1.5, 2.5]


@pytest.fixture
def duckdb_con() -> Iterator[Any]:
    con = xo.duckdb.connect()
    yield con
    con.disconnect()


def bundled_shape(shape: str, path: Path, con: Any) -> ir.Table:
    """One shape a bundled source takes, or all three in one build."""
    con.create_table("dt", RECORDED)
    shapes = {
        "memtable": lambda: xo.memtable(RECORDED.to_pandas(), name="mt"),
        "memory-backend-table": lambda: con.table("dt"),
        "relocated-read": lambda: deferred_read_parquet(path, con, table_name="r"),
    }
    if shape == "all":
        return toolz.reduce(ir.Table.union, (make() for make in shapes.values()))
    return shapes[shape]()


@pytest.mark.parametrize(
    "shape",
    tuple(
        pytest.param(shape, id=shape)
        for shape in ("memtable", "memory-backend-table", "relocated-read", "all")
    ),
)
def test_a_bundled_only_build_refreshes_like_a_plain_load(
    shape: str,
    tmp_path: Path,
    builds_dir: Path,
    duckdb_con: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift-exempt: never probed, so neither the cwd nor the original path is read.

    The cwd holds `GROWN` at every archive-relative path, so a bundled source
    resolved against it rather than the archive would move.
    """
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    build_path = build_expr(
        bundled_shape(shape, path, duckdb_con), builds_dir=builds_dir
    )
    path.unlink()
    decoy = tmp_path / "decoy"
    for bundled in build_path.glob("*/*.parquet"):
        (decoy / bundled.parent.name).mkdir(parents=True, exist_ok=True)
        write_parquet(decoy / bundled.parent.name / bundled.name, GROWN)
    monkeypatch.chdir(decoy)

    record = BuildRecord.from_build_dir(build_path)
    assert record.source_leaves and record.external_leaves == ()
    (refreshed, loaded) = (refresh_build(build_path), load_expr(build_path))
    assert refreshed.schema() == loaded.schema() == xo.schema(RECORDED.schema)
    assert get_expr_hash(refreshed) == get_expr_hash(loaded)
    assert set(refreshed.execute().columns) == set(RECORDED.column_names)


def test_a_refresh_leaves_bundled_sources_as_loaded(
    con: SqliteBackend, tmp_path: Path, builds_dir: Path
) -> None:
    """Only the drifted external source moves; every bundled node is kept."""
    path = tmp_path / "t.parquet"
    write_parquet(path, RECORDED)
    xcon = xo.connect()
    t = con.table("t").into_backend(xcon, "t_moved")
    read = deferred_read_parquet(path, xcon, table_name="r")
    mt = xo.memtable(RECORDED.to_pandas(), name="mt")
    build_path = build_expr(
        t.select("a").union(read.select("a")).union(mt.select("a")),
        builds_dir=builds_dir,
    )
    replace_table(con, GROWN)

    record = BuildRecord.from_build_dir(build_path)
    loaded = load_expr(build_path)
    refreshed = refresh_schemas(loaded, live_schemas(record, iter_leaf_reports(record)))
    sources = (ops.InMemoryTable, ops.DatabaseTable, Read)
    kept = set(walk_nodes(sources, loaded)) & set(walk_nodes(sources, refreshed))
    assert {(type(node), node.name) for node in kept} == {
        (ops.InMemoryTable, "mt"),
        (Read, "r"),
    }


@pytest.mark.parametrize(
    ("make_expr", "op_type"),
    (
        pytest.param(
            lambda t, cache: t.filter(t.a > 1).cache(cache=cache),
            CachedNode,
            id="cached-node",
        ),
        pytest.param(lambda t, cache: filter_through_a_tag(t), Tag, id="tag"),
        pytest.param(
            lambda t, cache: (m := t.into_backend(xo.connect(), "moved")).filter(
                m.a > 1
            ),
            RemoteTable,
            id="remote-table",
        ),
    ),
)
def test_a_stored_schema_follows_its_parent(
    con: SqliteBackend,
    tmp_path: Path,
    builds_dir: Path,
    make_expr: Callable,
    op_type: type,
) -> None:
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    build_path = build_expr(make_expr(con.table("t"), cache), builds_dir=builds_dir)
    replace_table(con, GROWN)

    expr = refresh_build(build_path)
    (node,) = walk_nodes(op_type, expr)
    assert dict(node.schema) == dict(con.table("t").schema())
    assert "c" in expr.schema()


def test_a_refreshed_remote_table_executes(
    con: SqliteBackend, builds_dir: Path
) -> None:
    moved = con.table("t").into_backend(xo.connect(), "moved")
    build_path = build_expr(moved.filter(moved.a > 1), builds_dir=builds_dir)
    replace_table(con, GROWN)

    assert list(refresh_build(build_path).execute()["c"]) == [2.5]


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


def test_a_csv_read_refreshes(tmp_path: Path, builds_dir: Path) -> None:
    """The schema `read_kwargs` instruction is rewritten too."""
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
    """Known limitation: a declared schema reads as drift."""
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


def test_a_schema_kwarg_bound_as_none_stays_unset(
    tmp_path: Path, builds_dir: Path
) -> None:
    """duckdb's `read_json` binds `columns=None`, and cannot take a `Schema`."""
    path = tmp_path / "t.json"
    path.write_text('{"a": 1}\n')
    build_path = build_expr(
        read_json(xo.duckdb.connect(), path, "t"),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    path.write_text('{"a": 1, "c": 2.5}\n')

    expr = refresh_build(build_path)
    (read,) = walk_nodes(Read, expr)
    assert dict(read.read_kwargs)[ReadKwarg.columns] is None
    assert list(expr.execute()["c"]) == [2.5]


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


def test_a_refresh_error_over_a_validation_error_survives_a_process_boundary() -> None:
    """ibis's `ValidationError`s overwrite `args`, so they cannot be unpickled."""
    with pytest.raises(ValidationError) as excinfo:
        ops.Mean(ops.Literal("x", dt.string))
    error = SchemaRefreshError("Mean", excinfo.value)
    error.__notes__ = ["while refreshing"]
    error.unpicklable = lambda: None

    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, SchemaRefreshError)
    assert restored.op_name == "Mean"
    assert isinstance(restored.cause, RefreshCause)
    assert restored.cause.type_name == type(excinfo.value).__name__
    assert restored.__notes__ == ["while refreshing"]
    assert not hasattr(restored, "unpicklable")
    assert str(restored) == str(error)


def test_an_unregistered_expr_bearing_op_is_refused(world_leaf: tuple) -> None:
    """`replace_nodes`'s tripwire: its drifted payload would stay stale."""

    class UnregisteredExprHolder(ops.Node):
        payload: Expr

    build_path, record, leaf = world_leaf
    live = {leaf_key(leaf, record): xo.schema(GROWN.schema)}
    node = UnregisteredExprHolder(payload=load_expr(build_path))

    with pytest.raises(ValueError, match="not registered in OPAQUE_SPECS"):
        refresh_schemas(node, live)


def test_an_unrecorded_expr_arg_of_a_registered_op_is_refused(
    world_leaf: tuple,
) -> None:
    class SneakyFlightExpr(FlightExpr):
        payload: Expr = None

    build_path, record, leaf = world_leaf
    live = {leaf_key(leaf, record): xo.schema(GROWN.schema)}
    con = xo.connect()
    t = con.register(RECORDED, "t")
    node = SneakyFlightExpr(
        name="sneaky",
        schema=t.schema(),
        source=con,
        input_expr=t,
        unbound_expr=xo.table(t.schema(), name="u"),
        make_server=toolz.identity,
        make_connection=toolz.identity,
        payload=load_expr(build_path),
    )

    with pytest.raises(ValueError, match="NON_EDGE_EXPR_FIELDS"):
        refresh_schemas(node, live)


def test_every_unmatched_key_is_named_and_labeled(world_leaf: tuple) -> None:
    build_path, record, leaf = world_leaf
    _, profile_key, _, recorded = leaf_key(leaf, record)
    live = {
        (str(kind), profile_key, name, recorded): xo.schema(GROWN.schema)
        for kind, name in (
            (LeafKind.DATABASE_TABLE, "not-a-table"),
            (LeafKind.READ, "not-a-read"),
        )
    }

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(load_expr(build_path), live)
    assert excinfo.value.op_name == "DatabaseTable, Read"
    assert "not-a-table" in str(excinfo.value)
    assert "not-a-read" in str(excinfo.value)


def test_one_key_both_equal_and_changed_is_refused(world_leaf: tuple) -> None:
    """The e2e case below covers changed-vs-changed."""
    _, record, leaf = world_leaf
    reports = (
        LeafReport(leaf, Verdict.CHANGED, live=xo.schema(GROWN.schema)),
        LeafReport(leaf, Verdict.EQUAL, live=leaf.recorded),
    )

    with pytest.raises(SchemaRefreshError, match="disagree on its live schema"):
        live_schemas(record, reports)


def test_two_reads_of_one_path_that_disagree_are_refused(
    tmp_path: Path, builds_dir: Path
) -> None:
    """`read_json`: csv/parquet inference ignores read options."""
    path = tmp_path / "t.json"
    path.write_text('{"a": 1}\n')
    con = xo.duckdb.connect()
    build_path = build_expr(
        read_json(con, path, "nested").union(
            read_json(con, path, "flat", maximum_depth=1)
        ),
        builds_dir=builds_dir,
        relocate_reads=False,
    )
    path.write_text('{"a": {"x": 1}}\n')
    record = BuildRecord.from_build_dir(build_path)
    reports = tuple(iter_leaf_reports(record))
    assert len(reports) == 2
    assert len({leaf_key(report.leaf, record) for report in reports}) == 1

    with pytest.raises(SchemaRefreshError) as excinfo:
        live_schemas(record, reports)
    assert excinfo.value.op_name == "Read"
    assert "disagree on its live schema" in str(excinfo.value)


def test_one_key_reported_twice_at_one_live_schema_refreshes(
    world_leaf: tuple,
) -> None:
    _, record, leaf = world_leaf
    report = LeafReport(leaf, Verdict.CHANGED, live=xo.schema(GROWN.schema))

    assert live_schemas(record, (report, report)) == {
        leaf_key(leaf, record): xo.schema(GROWN.schema)
    }


def test_a_source_that_went_empty_fails_its_dependents(world_leaf: tuple) -> None:
    """A zero-column schema is falsy but still applied."""
    build_path, record, leaf = world_leaf
    live = {leaf_key(leaf, record): xo.schema({})}

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(load_expr(build_path), live)
    assert excinfo.value.op_name == "Field"


def unprobeable_record(*paths: str) -> BuildRecord:
    """A union of sqlite-bound ``read_json`` reads: none is probeable."""
    nodes = {
        f"@read_{i}": {
            "op": "Read",
            "name": f"src{i}",
            "method_name": "read_json",
            "profile": "p0",
            "read_kwargs": [["hash_path", path], ["table_name", f"src{i}"]],
            "schema_ref": "schema_0",
        }
        for i, path in enumerate(paths)
    }
    nodes["@union"] = {
        "op": "Union",
        "values": [{"node_ref": ref} for ref in nodes],
        "distinct": False,
    }
    return BuildRecord(
        {
            "definitions": {
                "dtypes": {},
                "nodes": nodes,
                "schemas": {
                    "schema_0": {
                        "a": {"op": "DataType", "type": "Int64", "nullable": True}
                    }
                },
            },
            "expression": {"node_ref": "@union"},
        },
        {"p0": {"con_name": "sqlite"}},
    )


def test_every_unprobeable_source_is_named() -> None:
    record = unprobeable_record("/data/one.json", "/data/two.json")
    assert len(unchecked_leaves(record)) == 2

    with pytest.raises(SchemaRefreshError) as excinfo:
        check_refreshable(record)
    assert excinfo.value.op_name == "Read"
    assert "/data/one.json" in str(excinfo.value)
    assert "/data/two.json" in str(excinfo.value)


def test_a_checkable_build_is_refreshable(world_leaf: tuple) -> None:
    _, record, _ = world_leaf
    check_refreshable(record)


def test_an_expr_udf_rebinds_over_its_drifted_source(
    con: SqliteBackend, builds_dir: Path
) -> None:
    """`computed_kwargs_expr` is rebound by method, not recreated."""

    @udf.agg.pandas_df(schema=xo.schema({"a": "int64"}), return_type=dt.float64)
    def a_sum(frame):
        return frame["a"].astype(float).sum()

    t = con.table("t").into_backend(xo.connect(), "moved")
    add_sum = udf.make_pandas_expr_udf(
        computed_kwargs_expr=a_sum.on_expr(t).name("s").as_table(),
        fn=lambda value, frame, **kw: frame["x"] + float(value),
        schema=xo.schema({"x": dt.float64}),
        name="add_sum",
        return_type=dt.float64,
        post_process_fn=toolz.identity,
    )
    data = xo.memtable({"x": [1.0, 2.0]})
    build_path = build_expr(
        data.mutate(out=add_sum.on_expr(data)), builds_dir=builds_dir
    )
    replace_table(con, GROWN.set_column(0, "a", pa.array([10, 20], pa.int64())))

    expr = refresh_build(build_path)
    (op,) = walk_nodes(udf.ExprScalarUDF, expr)
    (remote,) = walk_nodes(RemoteTable, op.computed_kwargs_expr)
    assert "c" in remote.schema
    assert list(expr.execute()["out"]) == [31.0, 32.0]


def drift_the_table(expr: xo.Expr, schema: pa.Schema = GROWN.schema) -> dict:
    """The `refresh_schemas` mapping that moves `expr`'s one sqlite table."""
    (table,) = (
        node
        for node in walk_nodes(ops.DatabaseTable, expr)
        if type(node) is ops.DatabaseTable
    )
    return {op_key(table): xo.schema(schema)}


def test_a_flight_udxf_takes_the_schema_of_its_moved_input(
    con: SqliteBackend,
) -> None:
    """`FlightUDXF.__init__` does not derive its output schema."""
    expr = flight_udxf(
        con.table("t"),
        process_df=toolz.identity,
        maybe_schema_in=lambda schema: True,
        maybe_schema_out=lambda schema: xo.schema(dict(schema) | {"n": dt.int64}),
        con=xo.connect(),
    )

    refreshed = refresh_schemas(expr, drift_the_table(expr))
    (node,) = walk_nodes(FlightUDXF, refreshed)
    assert node.schema == node.udxf.calc_schema_out(node.input_expr.schema())
    assert "c" in node.schema
    assert "c" in refreshed.schema()


def test_a_flight_expr_follows_an_added_column(con: SqliteBackend) -> None:
    """`FlightExpr.__init__` skips the `unbound_expr` check `from_exprs` runs."""
    t = con.table("t")
    expr = flight_expr(t, xo.table(t.schema()), con=xo.connect())

    refreshed = refresh_schemas(expr, drift_the_table(expr))
    (node,) = walk_nodes(FlightExpr, refreshed)
    (unbound,) = walk_nodes(ops.UnboundTable, node.unbound_expr)
    assert unbound.schema == node.input_expr.schema()
    assert "c" in node.schema
    assert "c" in refreshed.schema()
    # `from_exprs` stores a cloudpickle round-trip; the rebuilt one must survive it
    assert "c" in cloudpickle.loads(cloudpickle.dumps(node.unbound_expr)).schema()


def test_a_flight_expr_over_a_dropped_column_names_the_field(
    con: SqliteBackend,
) -> None:
    t = con.table("t")
    expr = flight_expr(t, xo.table(t.schema()).select("a"), con=xo.connect())

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(expr, drift_the_table(expr, pa.schema({"b": pa.string()})))
    assert excinfo.value.op_name == "Field"


def test_a_flight_expr_without_an_unbound_table_is_named(
    con: SqliteBackend,
) -> None:
    t = con.table("t")
    expr = flight_expr(t, xo.table(t.schema()), con=xo.connect())
    (node,) = walk_nodes(FlightExpr, expr)
    unbound = recreate(node, unbound_expr=xo.memtable({"a": [1]})).to_expr()

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(unbound, drift_the_table(unbound))
    assert excinfo.value.op_name == "FlightExpr"
    assert "no UnboundTable" in str(excinfo.value)


def test_a_flight_source_keys_without_a_profile() -> None:
    with FlightServer(
        flight_url=make_flight_url(None),
        verify_client=False,
        make_connection=xo.duckdb.connect,
    ) as server:
        table = server.con.create_table("t", RECORDED)
        (kind, profile_key, name, _) = op_key(table.op())
    assert (kind, profile_key, name) == (LeafKind.DATABASE_TABLE, None, "t")
