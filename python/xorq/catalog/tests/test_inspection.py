"""Source-leaf extraction from a build record (xorq-labs/xorq#2294).

Every test here goes through the *record*, never through ``load_expr``: the
point of the module under test is that an entry whose expression can no longer
load still reports the sources it was built against.
"""

from __future__ import annotations

import datetime
import decimal
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog.catalog import Catalog, CatalogEntry
from xorq.catalog.enums import LeafKind, PinKey
from xorq.catalog.inspection import (
    BuildRecord,
    get_source_leaves,
    iter_source_leaves,
    reachable_node_refs,
)
from xorq.catalog.zip_utils import BuildZip
from xorq.ibis_yaml.enums import BundledSourceTypes, DumpFiles


EXOTIC_TABLE = pa.table(
    {
        "d": pa.array([decimal.Decimal("1.23")], pa.decimal128(10, 2)),
        "arr": pa.array([[1, 2]], pa.list_(pa.int64())),
        "ts": pa.array(
            [datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)],
            pa.timestamp("us", tz="UTC"),
        ),
    }
)


def make_world(root: Path) -> SimpleNamespace:
    """A sqlite table, a duckdb file database and two parquet files on disk."""
    root.mkdir(parents=True, exist_ok=True)
    src_path = root / "src.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}), src_path)
    exotic_path = root / "exotic.parquet"
    pq.write_table(EXOTIC_TABLE, exotic_path)
    db_path = root / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", pa.table({"a": [1, 2], "e": ["p", "q"]}).to_pandas())
    # duckdb is a memory_backend: its tables are materialized into the bundle at
    # build time, so this one stands in for every source that cannot drift.
    ddb_path = root / "live.ddb"
    ddb_con = xo.duckdb.connect(str(ddb_path))
    ddb_con.create_table("dt", pa.table({"a": [1, 2]}))
    return SimpleNamespace(
        root=root,
        src_path=src_path,
        exotic_path=exotic_path,
        db_path=db_path,
        con=con,
        ddb_path=ddb_path,
        ddb_con=ddb_con,
    )


def make_catalog(repo_path: Path) -> Catalog:
    return Catalog.from_repo_path(repo_path, init=True, annex=False)


def pandas_udf_expr(world: SimpleNamespace) -> xo.Expr:
    """A sqlite-sourced expression whose UDF lives as a pickled blob in a node def."""

    @xo.udf.make_pandas_udf(
        schema=xo.schema({"a": int}),
        return_type=xo.expr.datatypes.float64,
        name="dbl",
    )
    def dbl(df):
        return df["a"] * 2.0

    t = world.con.table("t")
    return t.mutate(result=dbl.on_expr(t))


def rewrite_member(
    zip_path: Path, member_name: str, rewrite: Callable[[bytes], bytes]
) -> None:
    """Rewrite one member of a build zip in place, leaving the rest byte-equal."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        contents = {info.filename: zf.read(info.filename) for info in zf.infolist()}
    (target,) = (name for name in contents if Path(name).name == member_name)
    contents[target] = rewrite(contents[target])
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in contents.items():
            zf.writestr(name, data)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    return make_world(tmp_path_factory.mktemp("inspection-world") / "world")


@pytest.fixture(scope="module")
def entries(
    tmp_path_factory: pytest.TempPathFactory, world: SimpleNamespace
) -> dict[str, CatalogEntry]:
    """One catalog entry per build shape, built once for the whole module."""
    catalog = make_catalog(tmp_path_factory.mktemp("inspection-catalog") / "repo")
    t = world.con.table("t")
    con = xo.connect()
    read = xo.deferred_read_parquet(world.src_path, con, table_name="src")
    exotic = xo.deferred_read_parquet(
        world.exotic_path, xo.connect(), table_name="exotic"
    )
    cache = ParquetCache.from_kwargs(
        source=xo.connect(), base_path=tmp_path_factory.mktemp("inspection-cache")
    )
    # relocate_reads=False so the pin's frozen read keeps its absolute cache-dir
    # path: an unbundled read is exactly the shape a false external leaf takes.
    pinned = read.filter(read.a > 1).cache(cache=cache).ls.pin(ensure_materialized=True)
    return {
        "sqlite": catalog.add(t.filter(t.a > 1)),
        "pinned": catalog.add(pinned, relocate_reads=False),
        "read_relocated": catalog.add(read.filter(read.a > 1)),
        "read_absolute": catalog.add(read.filter(read.a > 1), relocate_reads=False),
        "exotic": catalog.add(exotic, relocate_reads=False),
        "self_join": catalog.add(t.join(world.con.table("t").view(), "a")),
        "cached": catalog.add(t.into_backend(xo.duckdb.connect(), "t_remote").cache()),
        "mixed": catalog.add(read.join(t.into_backend(con, "t_mixed"), "a")),
        "bundled_only": catalog.add(
            world.ddb_con.table("dt")
            .select("a")
            .union(xo.memtable({"z": [1]}).select(a="z"))
        ),
    }


def test_database_table_leaf(entries: dict[str, CatalogEntry]) -> None:
    (leaf,) = get_source_leaves(entries["sqlite"])
    assert leaf.kind == LeafKind.DATABASE_TABLE
    assert leaf.name == "t"
    assert leaf.table == "t"
    assert leaf.namespace == (None, None)
    assert leaf.profile
    assert not leaf.bundled
    assert leaf.bundle_kind is None
    assert leaf.method_name is None
    assert dict(leaf.recorded) == dict(xo.schema({"a": "int64", "e": "string"}))


def test_database_table_display_name_is_dotted() -> None:
    """A DatabaseTable's name is its catalog/database/table path."""
    doc = {
        "definitions": {
            "dtypes": {},
            "nodes": {
                "@databasetable_0": {
                    "op": "DatabaseTable",
                    "table": "matches",
                    "profile": "p0",
                    "namespace": {"catalog": "warehouse", "database": "public"},
                    "schema_ref": "schema_0",
                }
            },
            "schemas": {
                "schema_0": {"x": {"op": "DataType", "type": "Int64", "nullable": True}}
            },
        },
        "expression": {"node_ref": "@databasetable_0"},
    }
    (leaf,) = iter_source_leaves(doc)
    assert leaf.name == "warehouse.public.matches"
    assert leaf.namespace == ("warehouse", "public")


def test_read_leaf_names_the_path_not_the_table(
    entries: dict[str, CatalogEntry], world: SimpleNamespace
) -> None:
    """A Read's display name is its recorded path, never the generated name."""
    (leaf,) = get_source_leaves(entries["read_absolute"])
    assert leaf.kind == LeafKind.READ
    assert leaf.name == str(world.src_path)
    assert leaf.table == "src"
    assert leaf.method_name == "read_parquet"


def test_relocation_decides_bundling(entries: dict[str, CatalogEntry]) -> None:
    """The same read is bundled when built by default, external when not."""
    (relocated,) = get_source_leaves(entries["read_relocated"])
    (absolute,) = get_source_leaves(entries["read_absolute"])
    assert relocated.bundled
    assert relocated.bundle_kind == BundledSourceTypes.read
    # a bundled read is named by its path inside the archive, not its source
    assert relocated.name == dict(relocated.read_kwargs)["read_path"]
    assert relocated.name.startswith(f"{BundledSourceTypes.read}/")
    assert not absolute.bundled
    assert absolute.bundle_kind is None
    # Same source, same recorded schema -- only the location differs.
    assert dict(relocated.recorded) == dict(absolute.recorded)


def test_recorded_schema_keeps_dtype_parameters(
    entries: dict[str, CatalogEntry],
) -> None:
    """Decimal, array and tz-aware timestamp survive with their parameters."""
    (leaf,) = get_source_leaves(entries["exotic"])
    recorded = dict(leaf.recorded)
    assert (recorded["d"].precision, recorded["d"].scale) == (10, 2)
    assert recorded["arr"].value_type.is_int64()
    assert recorded["ts"].timezone == "UTC"


def test_self_join_reports_one_leaf(entries: dict[str, CatalogEntry]) -> None:
    """The registry keys on content hash, so one table read twice is one source."""
    (leaf,) = get_source_leaves(entries["self_join"])
    assert (leaf.kind, leaf.name) == (LeafKind.DATABASE_TABLE, "t")


def test_cache_and_into_backend_report_only_true_sources(
    entries: dict[str, CatalogEntry],
) -> None:
    """CachedNode and RemoteTable subclass DatabaseTable; neither is a source."""
    record = BuildRecord.from_catalog_entry(entries["cached"])
    (leaf,) = record.source_leaves
    assert (leaf.kind, leaf.name) == (LeafKind.DATABASE_TABLE, "t")
    # the nodes an isinstance check would have swallowed really are in the record
    nodes = record.expr_doc["definitions"]["nodes"]
    ops = {nodes[ref]["op"] for ref in reachable_node_refs(record.expr_doc)}
    assert {"CachedNode", "RemoteTable"} <= ops


def test_pinned_cache_reports_no_external_leaf(
    entries: dict[str, CatalogEntry], world: SimpleNamespace
) -> None:
    """A pin's frozen read is a cache artifact, not a source, and the upstream it
    discarded is not this record's source either."""
    record = BuildRecord.from_catalog_entry(entries["pinned"])
    (leaf,) = record.source_leaves
    assert (leaf.kind, leaf.pinned, leaf.bundled) == (LeafKind.READ, True, False)
    assert leaf.drift_exempt
    assert record.external_leaves == ()

    nodes = record.expr_doc["definitions"]["nodes"]
    assert PinKey.OP in {node["op"] for node in nodes.values()}
    # the discarded upstream read is in the record, and out of the walk's reach
    hash_paths = {
        dict(map(tuple, node.get("read_kwargs", ()))).get("hash_path")
        for node in nodes.values()
    }
    assert str(world.src_path) in hash_paths
    assert leaf.name != str(world.src_path)


def test_bundled_only_entry_has_no_external_leaves(
    entries: dict[str, CatalogEntry],
) -> None:
    record = BuildRecord.from_catalog_entry(entries["bundled_only"])
    assert record.external_leaves == ()
    assert all(leaf.bundled for leaf in record.source_leaves)
    assert dict(record.bundled_counts) == {
        BundledSourceTypes.database_table: 1,
        BundledSourceTypes.inmemory: 1,
    }


def test_mixed_entry_keeps_only_the_external_leaf(
    entries: dict[str, CatalogEntry],
) -> None:
    """One live sqlite table plus one relocated read: the filter must split them."""
    record = BuildRecord.from_catalog_entry(entries["mixed"])
    assert len(record.source_leaves) == 2
    (external,) = record.external_leaves
    assert (external.kind, external.name) == (LeafKind.DATABASE_TABLE, "t")
    assert record.bundled_counts == ((BundledSourceTypes.read, 1),)


def test_leaf_order_is_stable(entries: dict[str, CatalogEntry]) -> None:
    record = BuildRecord.from_catalog_entry(entries["bundled_only"])
    node_refs = tuple(leaf.node_ref for leaf in record.source_leaves)
    assert node_refs == tuple(sorted(node_refs))
    assert record.source_leaves == get_source_leaves(entries["bundled_only"])


def test_reads_only_two_members_and_never_extracts(
    entries: dict[str, CatalogEntry], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two members, read in place: no tempdir, no extraction, no cleanup."""
    read_members = []
    original = BuildZip.read_member

    def spy(self, member_path, read_f):
        read_members.append(Path(member_path).name)
        return original(self, member_path, read_f)

    def no_extraction(*args, **kwargs):
        raise AssertionError("build archive must not be extracted")

    monkeypatch.setattr(BuildZip, "read_member", spy)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", no_extraction)
    monkeypatch.setattr(zipfile.ZipFile, "extract", no_extraction)

    assert get_source_leaves(entries["sqlite"])
    assert read_members == [DumpFiles.expr, DumpFiles.profiles]


def test_profile_is_resolvable_from_the_record(
    entries: dict[str, CatalogEntry],
) -> None:
    record = BuildRecord.from_catalog_entry(entries["sqlite"])
    (leaf,) = record.source_leaves
    assert record.get_profile_dict(leaf)["con_name"] == "sqlite"


def test_unloadable_entry_still_yields_leaves(
    tmp_path_factory: pytest.TempPathFactory, world: SimpleNamespace
) -> None:
    """The ticket's core claim: a record that cannot load still reports sources."""
    catalog = make_catalog(tmp_path_factory.mktemp("inspection-corrupt") / "repo")
    entry = catalog.add(pandas_udf_expr(world))

    def corrupt(data: bytes) -> bytes:
        (line,) = (
            line for line in data.splitlines() if line.strip().startswith(b"pickle:")
        )
        return data.replace(line, line.split(b"pickle:")[0] + b"pickle: AAAA")

    rewrite_member(entry.catalog_path, DumpFiles.expr, corrupt)

    with pytest.raises(pickle.UnpicklingError):
        entry.load_expr()

    (leaf,) = get_source_leaves(entry)
    assert (leaf.kind, leaf.name) == (LeafKind.DATABASE_TABLE, "t")


def test_extraction_never_unpickles(
    tmp_path_factory: pytest.TempPathFactory,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pandas UDF lives as a pickled blob in a node def; the walk never reaches it."""
    catalog = make_catalog(tmp_path_factory.mktemp("inspection-udf") / "repo")
    entry = catalog.add(pandas_udf_expr(world))

    def refuse(*args, **kwargs):
        raise AssertionError("source extraction must not unpickle")

    monkeypatch.setattr(cloudpickle, "loads", refuse)

    (leaf,) = get_source_leaves(entry)
    assert (leaf.kind, leaf.name) == (LeafKind.DATABASE_TABLE, "t")


def test_read_kwargs_are_deep_frozen_so_leaves_hash() -> None:
    """Dict-valued read_kwargs must not make a frozen leaf unhashable."""
    doc = {
        "definitions": {
            "dtypes": {},
            "nodes": {
                "@read_0": {
                    "op": "Read",
                    "name": "src",
                    "method_name": "read_csv",
                    "profile": "p0",
                    "read_kwargs": [
                        ["hash_path", "s3://bucket/src.csv"],
                        ["columns", {"x": "int64"}],
                    ],
                    "schema_ref": "schema_0",
                }
            },
            "schemas": {
                "schema_0": {"x": {"op": "DataType", "type": "Int64", "nullable": True}}
            },
        },
        "expression": {"node_ref": "@read_0"},
    }
    (leaf,) = iter_source_leaves(doc)
    assert dict(dict(leaf.read_kwargs)["columns"]) == {"x": "int64"}
    assert len({leaf}) == 1


def test_dangling_node_ref_names_the_ref() -> None:
    """A corrupt record fails with the missing ref, not a bare KeyError."""
    doc = {"definitions": {"nodes": {}}, "expression": {"node_ref": "@filter_0"}}
    with pytest.raises(ValueError, match="@filter_0"):
        reachable_node_refs(doc)


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        pytest.param(
            {"expression": {"node_ref": "@read_0"}},
            "has no definitions",
            id="no-definitions",
        ),
        pytest.param(
            {"definitions": {}, "expression": {"node_ref": "@read_0"}},
            "has no nodes",
            id="no-nodes",
        ),
        pytest.param(
            {"definitions": {"nodes": {}}}, "has no expression", id="no-expression"
        ),
    ],
)
def test_truncated_doc_never_reports_zero_sources(doc: dict, match: str) -> None:
    """A doc missing a required member must fail, not read as "no sources"."""
    with pytest.raises(ValueError, match=match):
        iter_source_leaves(doc)


def read_doc(**node_overrides: object) -> dict:
    """A minimal hand-built doc holding one reachable ``Read`` node def.

    An override of ``None`` drops that key from the node def, which is how a
    truncated node def is built.
    """
    node = {
        "op": "Read",
        "name": "src",
        "method_name": "read_parquet",
        "profile": "p0",
        "read_kwargs": [["hash_path", "s3://bucket/src.parquet"]],
        "schema_ref": "schema_0",
    } | node_overrides
    return {
        "definitions": {
            "dtypes": {},
            "nodes": {"@read_0": {k: v for k, v in node.items() if v is not None}},
            "schemas": {
                "schema_0": {"x": {"op": "DataType", "type": "Int64", "nullable": True}}
            },
        },
        "expression": {"node_ref": "@read_0"},
    }


def test_missing_schema_ref_names_the_node() -> None:
    """A leaf node def missing schema_ref fails with the ref, not a KeyError."""
    with pytest.raises(ValueError, match="@read_0"):
        iter_source_leaves(read_doc(schema_ref=None))


def test_unknown_registry_section_is_dropped() -> None:
    """A registry section a newer xorq added must not break extraction."""
    doc = read_doc()
    doc["definitions"]["future_section"] = {"x": 1}
    (leaf,) = iter_source_leaves(doc)
    assert leaf.kind == LeafKind.READ


def test_unknown_bundle_prefix_stays_bundled() -> None:
    """An unrecognized bundle dir degrades the kind, it does not kill the report."""
    doc = read_doc(read_kwargs=[["read_path", "future_bundle/src.parquet"]])
    (leaf,) = iter_source_leaves(doc)
    assert leaf.bundled
    assert leaf.bundle_kind is None


def test_bundled_counts_tolerates_unknown_kind() -> None:
    """A known and an unknown bundle kind sort together, they do not raise."""
    doc = read_doc(read_kwargs=[["read_path", "reads/known.parquet"]])
    node = doc["definitions"]["nodes"]["@read_0"]
    doc["definitions"]["nodes"]["@read_1"] = node | {
        "name": "other",
        "read_kwargs": [["read_path", "future_bundle/src.parquet"]],
    }
    doc["expression"] = {
        "left": {"node_ref": "@read_0"},
        "right": {"node_ref": "@read_1"},
    }
    record = BuildRecord(expr_doc=doc, profiles={})
    assert record.bundled_counts == ((BundledSourceTypes.read, 1), (None, 1))


def cache_tag_doc(live_upstream: bool) -> dict:
    """A pinned cache over one table, which a live branch optionally also reads."""
    expression = {"node_ref": "@cachetag_0"}
    if live_upstream:
        expression = {"left": expression, "right": {"node_ref": "@databasetable_0"}}
    return {
        "definitions": {
            "dtypes": {},
            "nodes": {
                "@read_0": {
                    "op": "Read",
                    "name": "cache_key",
                    "method_name": "read_parquet",
                    "profile": "p0",
                    "read_kwargs": [["hash_path", "/home/me/.cache/cache_key.parquet"]],
                    "schema_ref": "schema_0",
                },
                "@databasetable_0": {
                    "op": "DatabaseTable",
                    "table": "t",
                    "profile": "p1",
                    "schema_ref": "schema_0",
                },
                "@cachetag_0": {
                    "op": "CacheTag",
                    "parent": {"node_ref": "@read_0"},
                    "uncached": {"node_ref": "@databasetable_0"},
                    "cache": {"op": "ParquetCache"},
                    "schema_ref": "schema_0",
                },
            },
            "schemas": {
                "schema_0": {"x": {"op": "DataType", "type": "Int64", "nullable": True}}
            },
        },
        "expression": expression,
    }


def test_pin_hides_the_upstream_it_discarded() -> None:
    (leaf,) = iter_source_leaves(cache_tag_doc(live_upstream=False))
    assert (leaf.node_ref, leaf.pinned) == ("@read_0", True)


def test_a_leaf_shared_with_a_live_branch_stays_live() -> None:
    """`under_pin - live`: only a leaf reachable ONLY through the pin is pinned."""
    leaves = {leaf.node_ref: leaf for leaf in iter_source_leaves(cache_tag_doc(True))}
    assert leaves["@read_0"].pinned
    assert not leaves["@databasetable_0"].pinned
    record = BuildRecord(expr_doc=cache_tag_doc(True), profiles={})
    (external,) = record.external_leaves
    assert external.node_ref == "@databasetable_0"


def test_read_without_hash_path_falls_back_to_the_node_name() -> None:
    """The fallback is lazy: a present hash_path must not need `name` at all."""
    (leaf,) = iter_source_leaves(read_doc(read_kwargs=None))
    assert leaf.name == "src"
    (named,) = iter_source_leaves(read_doc(name=None))
    assert named.name == "s3://bucket/src.parquet"


def test_read_missing_both_path_and_name_names_the_node() -> None:
    with pytest.raises(ValueError, match="@read_0"):
        iter_source_leaves(read_doc(read_kwargs=None, name=None))


def test_missing_op_names_the_node() -> None:
    with pytest.raises(ValueError, match="@read_0"):
        iter_source_leaves(read_doc(op=None))


def test_dangling_profile_ref_names_the_profile() -> None:
    """A leaf naming a profile the record does not hold is a corrupt record."""
    record = BuildRecord(expr_doc=read_doc(), profiles={})
    (leaf,) = record.source_leaves
    with pytest.raises(ValueError, match="p0"):
        record.get_profile_dict(leaf)


def test_unparsable_expr_member_names_the_file(
    tmp_path_factory: pytest.TempPathFactory, world: SimpleNamespace
) -> None:
    """An empty expr.yaml must fail naming the member, not as an attrs TypeError."""
    catalog = make_catalog(tmp_path_factory.mktemp("inspection-empty") / "repo")
    entry = catalog.add(world.con.table("t").filter(xo._.a > 1))
    rewrite_member(entry.catalog_path, DumpFiles.expr, lambda data: b"")
    with pytest.raises(ValueError, match=DumpFiles.expr):
        get_source_leaves(entry)
