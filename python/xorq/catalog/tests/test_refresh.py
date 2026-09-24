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
import toolz

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog.drift import LeafReport, iter_leaf_reports, unchecked_leaves
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
from xorq.common.exceptions import InternalError, SchemaRefreshError
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
    make_read_kwargs,
    normalize_read_path_stat,
)
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import (
    CachedNode,
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


def test_a_retyped_case_base_fails_its_simple_case(
    con: SqliteBackend, builds_dir: Path
) -> None:
    """`SimpleCase` checks its base against each case in its constructor, not
    its signature; that rejection is drift too."""
    t = con.table("t")
    build_path = build_expr(
        t.mutate(c=t.a.cases((1, "one"), else_="other")), builds_dir=builds_dir
    )
    recreate(con, pa.table({"a": ["1", "x"], "b": ["x", "y"]}))

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == "SimpleCase"


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


def test_one_table_name_on_two_connections_is_two_sources(
    con: SqliteBackend, tmp_path: Path, builds_dir: Path
) -> None:
    """Same backend, table name and schema on two files: only the drifted one moves."""
    other = SqliteBackend().connect(str(tmp_path / "other.sqlite"))
    other.create_table("t", RECORDED.to_pandas())
    t, u = con.table("t"), other.table("t")
    remote = u.select(k=u.a, v=u.b).into_backend(con)
    build_path = build_expr(t.join(remote, t.a == remote.k), builds_dir=builds_dir)
    recreate(con, GROWN)

    schemas = {
        node.source._profile.kwargs_dict["database"]: node.schema
        for node in walk_nodes(ops.DatabaseTable, refresh_build(build_path))
        if type(node) is ops.DatabaseTable
    }
    assert "c" in schemas[str(tmp_path / "live.sqlite")]
    assert "c" not in schemas[str(tmp_path / "other.sqlite")]


def test_a_missing_source_stops_before_loading(world: tuple) -> None:
    con, build_path = world
    con.drop_table("t")

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_build(build_path)
    assert excinfo.value.op_name == "DatabaseTable"


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


def test_a_bug_in_the_rewrite_is_not_labeled_as_drift(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an op rejecting its new inputs becomes a `SchemaRefreshError`; a
    failure of the rewrite itself would otherwise send the user to their data."""
    con, build_path = world
    recreate(con, GROWN)

    def broken(node, **kwargs):
        raise AttributeError("a bug in the rewrite")

    monkeypatch.setattr("xorq.catalog.refresh.recreate", broken)
    with pytest.raises(AttributeError, match="a bug in the rewrite"):
        refresh_build(build_path)


def test_an_internal_error_in_the_rewrite_is_not_labeled_as_drift(
    world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An `InternalError` is a `XorqError`, but a bug all the same."""
    con, build_path = world
    recreate(con, GROWN)

    def broken(node, **kwargs):
        raise InternalError("a bug in the rewrite")

    monkeypatch.setattr("xorq.catalog.refresh.recreate", broken)
    with pytest.raises(InternalError, match="a bug in the rewrite") as excinfo:
        refresh_build(build_path)
    assert not isinstance(excinfo.value, SchemaRefreshError)


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


def test_a_key_that_matches_no_source_raises(world: tuple) -> None:
    """A drifted source the rewrite cannot find would otherwise keep its
    recorded schema while the refresh reports success."""
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    kind, con_name, _, recorded = leaf_key(leaf, record)
    live = {(kind, con_name, "not-a-table", recorded): xo.schema(GROWN.schema)}

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(load_expr(build_path), live)
    assert excinfo.value.op_name == "DatabaseTable"
    assert "not-a-table" in str(excinfo.value)


def test_every_key_that_matches_no_source_is_named(world: tuple) -> None:
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    kind, con_name, _, recorded = leaf_key(leaf, record)
    live = {
        (kind, con_name, name, recorded): xo.schema(GROWN.schema)
        for name in ("not-a-table", "nor-this-one")
    }

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(load_expr(build_path), live)
    assert excinfo.value.op_name == "DatabaseTable"
    assert "not-a-table" in str(excinfo.value)
    assert "nor-this-one" in str(excinfo.value)


def test_unmatched_keys_of_mixed_kinds_are_labeled_with_each(world: tuple) -> None:
    """The error names every key, so its label cannot be the first key's kind."""
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    _, con_name, _, recorded = leaf_key(leaf, record)
    live = {
        (str(LeafKind.DATABASE_TABLE), con_name, "not-a-table", recorded): xo.schema(
            GROWN.schema
        ),
        (str(LeafKind.READ), con_name, "not-a-read", recorded): xo.schema(GROWN.schema),
    }

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(load_expr(build_path), live)
    assert excinfo.value.op_name == "DatabaseTable, Read"


@pytest.mark.parametrize(
    "other",
    (
        pytest.param(Verdict.EQUAL, id="equal"),
        pytest.param(Verdict.CHANGED, id="changed"),
    ),
)
def test_one_key_at_two_live_schemas_is_refused(world: tuple, other: Verdict) -> None:
    """Two reads of one path with different options share a `leaf_key`; a
    mapping holding one of their live schemas would rebuild both over it."""
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    other_live = leaf.recorded if other == Verdict.EQUAL else xo.schema({"a": "int64"})
    reports = (
        LeafReport(leaf, Verdict.CHANGED, live=xo.schema(GROWN.schema)),
        LeafReport(leaf, other, live=other_live),
    )

    with pytest.raises(SchemaRefreshError) as excinfo:
        live_schemas(record, reports)
    assert excinfo.value.op_name == "DatabaseTable"
    assert "disagree on its live schema" in str(excinfo.value)


def test_two_reads_of_one_path_that_disagree_are_refused(
    tmp_path: Path, builds_dir: Path
) -> None:
    """The sweep end to end: two reads of one file that differ only in their
    options share a `leaf_key` and are probed as two leaves. `read_json` because
    csv and parquet are probed by an inference that ignores the read's options,
    so only a replayed read can come back at two schemas."""
    path = tmp_path / "t.json"
    path.write_text('{"a": 1}\n')
    con = xo.duckdb.connect()
    recorded = xo.schema({"a": "int64"})

    def read_json(name: str, **kwargs) -> ir.Table:
        return Read(
            method_name="read_json",
            name=name,
            schema=recorded,
            source=con,
            read_kwargs=make_read_kwargs(
                con.read_json, str(path), table_name=name, **kwargs
            ),
            normalize_method=normalize_read_path_stat,
        ).to_expr()

    build_path = build_expr(
        read_json("nested").union(read_json("flat", maximum_depth=1)),
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


def test_one_key_reported_twice_at_one_live_schema_refreshes(world: tuple) -> None:
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    report = LeafReport(leaf, Verdict.CHANGED, live=xo.schema(GROWN.schema))

    assert live_schemas(record, (report, report)) == {
        leaf_key(leaf, record): xo.schema(GROWN.schema)
    }


def test_a_source_that_went_empty_fails_its_dependents(world: tuple) -> None:
    """A zero-column schema is falsy; it still has to reach its dependents,
    and the Field that loses its column is the evidence that it did."""
    _, build_path = world
    record = BuildRecord.from_build_dir(build_path)
    (leaf,) = record.external_leaves
    live = {leaf_key(leaf, record): xo.schema({})}

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(load_expr(build_path), live)
    assert excinfo.value.op_name == "Field"


def unprobeable_record(*paths: str) -> BuildRecord:
    """A build unioning one ``read_json`` per path, bound to sqlite: reads with
    no registered inference on an ingesting backend, so none is ever probed.
    Only the walk from ``expression`` reads the union, so its args are minimal."""
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


def test_an_unprobeable_source_stops_the_refresh() -> None:
    """A read with no registered inference, bound to an ingesting backend, is
    never probed, so a refresh cannot vouch for its schema."""
    with pytest.raises(SchemaRefreshError) as excinfo:
        check_refreshable(unprobeable_record("/data/src.json"))
    assert excinfo.value.op_name == "Read"
    assert "/data/src.json" in str(excinfo.value)


def test_every_unprobeable_source_is_named() -> None:
    record = unprobeable_record("/data/one.json", "/data/two.json")
    assert len(unchecked_leaves(record)) == 2

    with pytest.raises(SchemaRefreshError) as excinfo:
        check_refreshable(record)
    assert excinfo.value.op_name == "Read"
    assert "/data/one.json" in str(excinfo.value)
    assert "/data/two.json" in str(excinfo.value)


def test_a_checkable_build_is_refreshable(world: tuple) -> None:
    _, build_path = world
    check_refreshable(BuildRecord.from_build_dir(build_path))


def test_a_remote_table_follows_its_remote_expr(
    con: SqliteBackend, builds_dir: Path
) -> None:
    t = con.table("t")
    moved = t.into_backend(xo.connect(), "moved")
    build_path = build_expr(moved.filter(moved.a > 1), builds_dir=builds_dir)
    recreate(con, GROWN)

    expr = refresh_build(build_path)
    (remote,) = walk_nodes(RemoteTable, expr)
    assert dict(remote.schema) == dict(con.table("t").schema())
    assert "c" in expr.schema()
    assert list(expr.execute()["c"]) == [2.5]


def test_an_expr_udf_rebinds_over_its_drifted_source(
    con: SqliteBackend, builds_dir: Path
) -> None:
    """`computed_kwargs_expr` sits in `__config__`, so it is rebound by method
    rather than recreated as a kwarg."""

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
    recreate(con, GROWN.set_column(0, "a", pa.array([10, 20], pa.int64())))

    expr = refresh_build(build_path)
    (op,) = walk_nodes(udf.ExprScalarUDF, expr)
    (remote,) = walk_nodes(RemoteTable, op.computed_kwargs_expr)
    assert "c" in remote.schema
    assert list(expr.execute()["out"]) == [31.0, 32.0]


def drift_the_table(expr: xo.Expr) -> dict:
    """The `refresh_schemas` mapping that grows `expr`'s one sqlite table."""
    (table,) = (
        node
        for node in walk_nodes(ops.DatabaseTable, expr)
        if type(node) is ops.DatabaseTable
    )
    return {op_key(table): xo.schema(GROWN.schema)}


def test_a_flight_udxf_takes_the_schema_of_its_moved_input(
    con: SqliteBackend,
) -> None:
    """`FlightUDXF.__init__` does not derive its output schema; a plain
    recreate would keep the one computed over the recorded input."""
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


def test_a_flight_expr_whose_input_no_longer_fits_raises(
    con: SqliteBackend,
) -> None:
    """`FlightExpr.__init__` skips the `unbound_expr` check `from_exprs` runs."""
    t = con.table("t")
    expr = flight_expr(t, xo.table(t.schema()).select("a"), con=xo.connect())

    with pytest.raises(SchemaRefreshError) as excinfo:
        refresh_schemas(expr, drift_the_table(expr))
    assert excinfo.value.op_name == "FlightExpr"


def test_a_lazy_load_connects_only_the_drifted_source(
    con: SqliteBackend, tmp_path: Path, builds_dir: Path
) -> None:
    """A source that did not drift is never connected, so one that can no
    longer connect does not stop the refresh."""
    other_path = tmp_path / "other.sqlite"
    other = SqliteBackend().connect(str(other_path))
    other.create_table("u", RECORDED.to_pandas())
    t, u = con.table("t"), other.table("u")
    remote = u.select(k=u.a, v=u.b).into_backend(con)
    build_path = build_expr(t.join(remote, t.a == remote.k), builds_dir=builds_dir)
    recreate(con, GROWN)
    record = BuildRecord.from_build_dir(build_path)
    live = live_schemas(record, iter_leaf_reports(record))
    # A directory where the database was: connecting to it raises.
    other.con.close()
    other_path.unlink()
    other_path.mkdir()

    refreshed = refresh_schemas(load_expr(build_path, lazy=True), live)
    (after_t,) = (n for n in walk_nodes(ops.DatabaseTable, refreshed) if n.name == "t")
    assert "c" in after_t.schema


def test_a_flight_source_keys_without_a_profile() -> None:
    with FlightServer(
        flight_url=make_flight_url(None),
        verify_client=False,
        make_connection=xo.duckdb.connect,
    ) as server:
        table = server.con.create_table("t", RECORDED)
        (kind, profile_key, name, _) = op_key(table.op())
    assert (kind, profile_key, name) == (LeafKind.DATABASE_TABLE, None, "t")
