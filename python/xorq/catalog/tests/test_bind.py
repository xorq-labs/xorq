from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.catalog.backend import GitBackend
from xorq.catalog.bind import (
    CatalogTag,
    _make_source_expr,
    _validate_schema,
    bind,
    fuse_catalog_source,
)
from xorq.catalog.catalog import Catalog
from xorq.catalog.composer import ExprComposer
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import HashingTag, Read, RemoteTable
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.types.core import ExprMetadata


@pytest.fixture
def catalog(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("bind-repo"))
    return Catalog(backend=GitBackend(repo=repo))


@pytest.fixture
def source_expr():
    return xo.memtable(
        {"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0], "name": ["a", "b", "c"]}
    )


@pytest.fixture
def transform_expr(source_expr):
    """An unbound transform that filters on amount > 0."""
    schema = source_expr.schema()
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    return unbound.filter(unbound.amount > 0).select("user_id", "amount")


@pytest.fixture
def catalog_with_entries(catalog, source_expr, transform_expr):
    """Catalog populated with a source and an unbound transform."""
    source_entry = catalog.add(source_expr, aliases=("my-source",))
    transform_entry = catalog.add(transform_expr, aliases=("my-transform",))
    return catalog, source_entry, transform_entry


@pytest.fixture
def catalog_with_bound(catalog_with_entries):
    """Catalog with source, transform, bound entry, and a second transform."""
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    bound_entry = catalog.add(bound, aliases=("bound1",))

    output_schema = bound.as_table().schema()
    unbound2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    transform2 = unbound2.filter(unbound2.amount > 15)
    transform2_entry = catalog.add(transform2, aliases=("transform2",))
    return catalog, bound_entry, transform2_entry


# --- ExprKind tests ---


def test_source_kind(source_expr):
    meta = ExprMetadata.from_expr(source_expr)
    assert meta.kind == ExprKind.Source


def test_unbound_kind(transform_expr):
    meta = ExprMetadata.from_expr(transform_expr)
    assert meta.kind == ExprKind.UnboundExpr
    assert meta.schema_in is not None


def test_bound_kind(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    meta = ExprMetadata.from_expr(bound)
    assert meta.kind == ExprKind.Composed
    assert len(meta.composed_from) == 2
    source_entries = [s for s in meta.composed_from if s["kind"] == "source"]
    assert len(source_entries) == 1


def test_bound_to_dict_includes_composed_from(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    meta = ExprMetadata.from_expr(bound)
    d = meta.to_dict()
    assert d["kind"] == "composed"
    assert "composed_from" in d
    assert len(d["composed_from"]) == 2


def test_chained_bind_kind(catalog_with_bound):
    catalog, bound_entry, transform2_entry = catalog_with_bound
    bound2 = bind(bound_entry, transform2_entry)
    meta = ExprMetadata.from_expr(bound2)
    assert meta.kind == ExprKind.Composed
    assert len(meta.composed_from) >= 1


# --- Schema validation tests ---


def test_validate_schema_exact_match():
    s = Schema({"a": "int64", "b": "string"})
    _validate_schema(s, s, "src", "trn")


def test_validate_schema_superset_match():
    source = Schema({"a": "int64", "b": "string", "c": "float64"})
    transform = Schema({"a": "int64", "b": "string"})
    _validate_schema(source, transform, "src", "trn")


def test_validate_schema_missing_column():
    source = Schema({"a": "int64"})
    transform = Schema({"a": "int64", "b": "string"})
    with pytest.raises(ValueError, match="missing"):
        _validate_schema(source, transform, "src", "trn")


def test_validate_schema_type_mismatch():
    source = Schema({"a": "int64", "b": "string"})
    transform = Schema({"a": "int64", "b": "float64"})
    with pytest.raises(ValueError, match="type mismatch"):
        _validate_schema(source, transform, "src", "trn")


def test_validate_schema_missing_and_type_mismatch():
    source = Schema({"a": "int64", "b": "string"})
    transform = Schema({"a": "float64", "c": "int64"})
    with pytest.raises(ValueError, match="missing") as exc_info:
        _validate_schema(source, transform, "src", "trn")
    assert "type mismatch" in str(exc_info.value)


# --- Bind tests ---


def test_bind_produces_expr(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    assert bound is not None
    meta = ExprMetadata.from_expr(bound)
    assert meta.kind == ExprKind.Composed


def test_bind_not_unbound_raises(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    with pytest.raises(ValueError, match="no UnboundTable"):
        bind(source_entry, source_entry)


def test_bind_roundtrip_catalog(catalog_with_entries):
    """Bound entry can be added to catalog and loaded back."""
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    bound_entry = catalog.add(bound, aliases=("bound-result",))
    assert bound_entry.kind == ExprKind.Composed
    assert len(bound_entry.composed_from) == 2


def test_bind_with_alias(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry, alias="custom-alias")
    meta = ExprMetadata.from_expr(bound)
    source_sources = tuple(s for s in meta.composed_from if s["kind"] == "source")
    assert source_sources[0]["alias"] == "custom-alias"


def test_bind_bound_entry_as_source(catalog_with_bound):
    """A bound entry can be used as the source for another bind."""
    catalog, bound_entry, transform2_entry = catalog_with_bound
    bound2 = bind(bound_entry, transform2_entry)
    assert bound2 is not None
    meta = ExprMetadata.from_expr(bound2)
    assert meta.kind == ExprKind.Composed


def test_bind_bound_entry_executes(catalog_with_bound):
    """Binding a bound entry produces an executable expression."""
    catalog, bound_entry, transform2_entry = catalog_with_bound
    bound2 = bind(bound_entry, transform2_entry)
    result = bound2.execute()
    assert len(result) == 2
    assert set(result["user_id"]) == {2, 3}


def test_bind_bound_roundtrip_catalog(catalog_with_bound):
    """Chained bind can be added to catalog and loaded back."""
    catalog, bound_entry, transform2_entry = catalog_with_bound
    bound2 = bind(bound_entry, transform2_entry)
    bound2_entry = catalog.add(bound2, aliases=("bound2",))
    assert bound2_entry.kind == ExprKind.Composed
    assert len(bound2_entry.composed_from) >= 1


def test_bind_schema_mismatch(catalog):
    """Binding incompatible schemas raises ValueError."""
    source = xo.memtable({"x": [1, 2], "y": ["a", "b"]})
    schema = xo.Schema({"a": "int64", "b": "float64"})
    unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
    transform = unbound.filter(unbound.a > 0)

    source_entry = catalog.add(source)
    transform_entry = catalog.add(transform)

    with pytest.raises(ValueError, match="mismatch"):
        bind(source_entry, transform_entry)


def test_bind_variadic(catalog_with_entries):
    """bind(source, t1, t2) chains all transforms in one call."""
    catalog, source_entry, transform_entry = catalog_with_entries

    output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15), aliases=("t2",))

    bound = bind(source_entry, transform_entry, t2_entry)
    result = bound.execute()
    assert len(result) == 2
    assert set(result["user_id"]) == {2, 3}


def test_bind_no_transforms_raises(catalog_with_entries):
    """bind() with zero transforms raises ValueError."""
    _, source_entry, _ = catalog_with_entries
    with pytest.raises(ValueError, match="At least one transform"):
        bind(source_entry)


def test_bind_plain_expr_as_transform_raises(catalog_with_entries):
    """Using a source (no UnboundTable) as transform raises ValueError."""
    catalog, source_entry, _ = catalog_with_entries
    another_source = xo.memtable({"user_id": [4], "amount": [40.0]})
    another_entry = catalog.add(another_source, aliases=("another-source",))
    with pytest.raises(ValueError, match="no UnboundTable"):
        bind(source_entry, another_entry)


def test_bind_cross_catalog_raises(catalog_with_entries, tmpdir):
    """Binding entries from different catalogs raises ValueError."""
    catalog, source_entry, _ = catalog_with_entries
    other_repo = Catalog.init_repo_path(Path(tmpdir).joinpath("other-repo"))
    other_catalog = Catalog(backend=GitBackend(repo=other_repo))
    other_transform = xo.memtable({"user_id": [1], "amount": [10.0]})
    schema = other_transform.schema()
    unbound = ops.UnboundTable(name="ph", schema=schema).to_expr()
    transform = unbound.filter(unbound.amount > 0)
    other_entry = other_catalog.add(transform)
    with pytest.raises(ValueError, match="Got multiple catalogs"):
        bind(source_entry, other_entry)


# --- get_catalog_entry tests ---


def test_get_catalog_entry_by_name(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    resolved = catalog.get_catalog_entry(source_entry.name)
    assert resolved.name == source_entry.name


def test_get_catalog_entry_by_alias(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    resolved = catalog.get_catalog_entry("my-source", maybe_alias=True)
    assert resolved.name == source_entry.name


def test_get_catalog_entry_unknown_raises(catalog_with_entries):
    catalog, _, _ = catalog_with_entries
    with pytest.raises(ValueError, match="not found"):
        catalog.get_catalog_entry("nonexistent", maybe_alias=True)


# --- catalog.load() tests ---


def test_catalog_source_returns_catalog_source_expr(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    expr = catalog.load("my-source")
    assert expr is not None
    meta = ExprMetadata.from_expr(expr)
    assert meta.kind == ExprKind.Composed
    assert len(meta.composed_from) == 1
    assert meta.composed_from[0]["kind"] == "source"
    assert meta.composed_from[0]["entry_name"] == source_entry.name


def test_catalog_source_by_name(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    expr = catalog.load(source_entry.name)
    meta = ExprMetadata.from_expr(expr)
    assert meta.composed_from[0]["kind"] == "source"


def test_catalog_source_executes(catalog_with_entries):
    catalog, _, _ = catalog_with_entries
    expr = catalog.load("my-source")
    result = expr.execute()
    assert len(result) == 3


# --- catalog.bind() tests ---


def test_catalog_bind_produces_composed(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = catalog.bind(source_entry, transform_entry)
    meta = ExprMetadata.from_expr(bound)
    assert meta.kind == ExprKind.Composed
    assert len(meta.composed_from) == 2


def test_catalog_bind_source_provenance(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = catalog.bind(source_entry, transform_entry)
    meta = ExprMetadata.from_expr(bound)

    source_entries = tuple(s for s in meta.composed_from if s["kind"] == "source")
    assert len(source_entries) == 1
    assert source_entries[0]["entry_name"] == source_entry.name


def test_catalog_bind_has_source_and_transform_tags(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = catalog.bind(source_entry, transform_entry)
    meta = ExprMetadata.from_expr(bound)

    source_entries = [s for s in meta.composed_from if s["kind"] == "source"]
    transform_entries = [s for s in meta.composed_from if s["kind"] == "unbound_expr"]
    assert len(source_entries) == 1
    assert source_entries[0]["entry_name"] == source_entry.name
    assert len(transform_entries) == 1
    assert transform_entries[0]["entry_name"] == transform_entry.name


def test_catalog_bind_executes(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = catalog.bind(source_entry, transform_entry)
    result = bound.execute()
    assert len(result) == 3
    assert set(result.columns) == {"user_id", "amount"}


def test_catalog_bind_not_unbound_raises(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    with pytest.raises(ValueError, match="no UnboundTable"):
        catalog.bind(source_entry, source_entry)


def test_catalog_bind_roundtrip_catalog(catalog_with_entries):
    """bind() result can be added to catalog and loaded back with kind info."""
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = catalog.bind(source_entry, transform_entry)
    bound_entry = catalog.add(bound, aliases=("bound-result",))
    assert bound_entry.kind == ExprKind.Composed
    assert len(bound_entry.composed_from) == 2


def test_catalog_bind_variadic(catalog_with_entries):
    """catalog.bind(source, t1, t2) chains transforms."""
    catalog, source_entry, transform_entry = catalog_with_entries

    output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15), aliases=("t2",))

    bound = catalog.bind(source_entry, transform_entry, t2_entry)
    result = bound.execute()
    assert len(result) == 2
    assert set(result["user_id"]) == {2, 3}


# --- Inline chaining tests ---


def test_bind_expr_as_source(catalog_with_entries):
    """bind accepts an expr (from a previous bind) as the source."""
    catalog, source_entry, transform_entry = catalog_with_entries
    step1 = bind(source_entry, transform_entry)

    output_schema = step1.as_table().schema()
    unbound2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    transform2 = unbound2.filter(unbound2.amount > 15)
    transform2_entry = catalog.add(transform2, aliases=("t2",))

    expr = bind(step1, transform2_entry)
    result = expr.execute()
    assert len(result) == 2
    assert set(result["user_id"]) == {2, 3}


def test_chain_three_steps(catalog_with_entries):
    """Three-step inline chain without intermediate catalog adds."""
    catalog, source_entry, transform_entry = catalog_with_entries

    schema2 = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=schema2).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15), aliases=("t2",))

    schema3 = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub3 = ops.UnboundTable(name="ph3", schema=schema3).to_expr()
    t3_entry = catalog.add(ub3.select("user_id"), aliases=("t3",))

    # All in one call
    result = bind(source_entry, transform_entry, t2_entry, t3_entry).execute()
    assert list(result.columns) == ["user_id"]
    assert set(result["user_id"]) == {2, 3}


def test_chain_has_source_and_all_transform_tags(catalog_with_entries):
    """Chained bind tags source and every transform step."""
    catalog, source_entry, transform_entry = catalog_with_entries

    schema2 = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=schema2).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15), aliases=("t2",))

    bound = bind(source_entry, transform_entry, t2_entry)
    meta = ExprMetadata.from_expr(bound)
    source_entries = [s for s in meta.composed_from if s["kind"] == "source"]
    transform_entries = [s for s in meta.composed_from if s["kind"] == "unbound_expr"]
    assert len(source_entries) == 1
    assert len(transform_entries) == 2


# --- Kind YAML roundtrip test ---


def test_kind_survives_yaml_roundtrip(catalog_with_entries):
    """kind field is preserved through build/load (YAML serialization)."""
    catalog, source_entry, transform_entry = catalog_with_entries
    bound = catalog.bind(source_entry, transform_entry)

    bound_entry = catalog.add(bound, aliases=("roundtrip-test",))

    loaded_expr = bound_entry.expr
    meta = ExprMetadata.from_expr(loaded_expr)

    assert meta.kind == ExprKind.Composed
    assert len(meta.composed_from) >= 1


# --- ExprComposer tests ---


def test_composed_expr_single_transform(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    composed = ExprComposer(source=source_entry, transforms=(transform_entry,))
    result = composed.expr.execute()
    assert len(result) == 3
    assert set(result.columns) == {"user_id", "amount"}


def test_composed_expr_has_transform_tag(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    composed = ExprComposer(source=source_entry, transforms=(transform_entry,))
    tags = walk_nodes(HashingTag, composed.expr)
    transform_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.TRANSFORM]
    assert len(transform_tags) == 1
    assert transform_tags[0].metadata["entry_name"] == transform_entry.name
    assert transform_tags[0].metadata["kind"] == str(transform_entry.kind)


def test_composed_expr_preserves_source_tag(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    composed = ExprComposer(source=source_entry, transforms=(transform_entry,))
    tags = walk_nodes(HashingTag, composed.expr)
    source_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.SOURCE]
    assert len(source_tags) == 1
    assert source_tags[0].metadata["entry_name"] == source_entry.name


def test_composed_expr_variadic_transforms(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15))

    composed = ExprComposer(source=source_entry, transforms=(transform_entry, t2_entry))
    result = composed.expr.execute()
    assert len(result) == 2
    assert set(result["user_id"]) == {2, 3}


def test_composed_expr_multiple_tags_ordered(catalog_with_entries):
    """Transform tags appear outermost-first: last-applied transform is first."""
    catalog, source_entry, transform_entry = catalog_with_entries
    output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15))

    composed = ExprComposer(source=source_entry, transforms=(transform_entry, t2_entry))
    tags = walk_nodes(HashingTag, composed.expr)
    transform_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.TRANSFORM]
    assert len(transform_tags) == 2
    # DFS from root finds outermost (last applied) tag first
    assert transform_tags[0].metadata["entry_name"] == t2_entry.name
    assert transform_tags[1].metadata["entry_name"] == transform_entry.name


def test_composed_expr_with_code(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    composed = ExprComposer(
        source=source_entry,
        transforms=(transform_entry,),
        code="source.filter(source.amount > 15)",
    )
    result = composed.expr.execute()
    assert len(result) == 2
    assert set(result["user_id"]) == {2, 3}


def test_composed_expr_code_tag(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    composed = ExprComposer(
        source=source_entry,
        transforms=(transform_entry,),
        code="source.filter(source.amount > 15)",
    )
    tags = walk_nodes(HashingTag, composed.expr)
    code_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.CODE]
    transform_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.TRANSFORM]
    assert len(code_tags) == 1
    assert code_tags[0].metadata["code"] == "source.filter(source.amount > 15)"
    assert len(transform_tags) == 1


def test_composed_expr_code_only(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    composed = ExprComposer(
        source=source_entry,
        code="source.filter(source.amount > 15)",
    )
    result = composed.expr.execute()
    assert len(result) == 2


def test_composed_expr_code_only_has_tag(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    composed = ExprComposer(
        source=source_entry,
        code="source.filter(source.amount > 15)",
    )
    tags = walk_nodes(HashingTag, composed.expr)
    code_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.CODE]
    source_tags = [t for t in tags if t.metadata.get("tag") == CatalogTag.SOURCE]
    assert len(code_tags) == 1
    assert len(source_tags) == 1


def test_composed_expr_bare_source(catalog_with_entries):
    _, source_entry, _ = catalog_with_entries
    composed = ExprComposer(source=source_entry)
    result = composed.expr.execute()
    assert len(result) > 0


def test_composed_expr_bad_source_raises():
    with pytest.raises(TypeError, match="must be.*CatalogEntry"):
        ExprComposer(source="not-an-entry", transforms=("also-bad",))


# --- ExprComposer.from_expr tests ---


def test_from_expr_single_transform(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    original = ExprComposer(source=source_entry, transforms=(transform_entry,))
    recovered = ExprComposer.from_expr(original.expr, catalog)

    assert recovered.source.name == source_entry.name
    assert len(recovered.transforms) == 1
    assert recovered.transforms[0].name == transform_entry.name
    assert recovered.code is None
    # alias resolves to the entry's first alias even when not explicitly passed
    assert recovered.alias == "my-source"


def test_from_expr_bare_source(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    original = ExprComposer(source=source_entry)
    recovered = ExprComposer.from_expr(original.expr, catalog)

    assert recovered.source.name == source_entry.name
    assert recovered.transforms == ()
    assert recovered.code is None


def test_from_expr_with_code(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    code = "source.filter(source.amount > 15)"
    original = ExprComposer(
        source=source_entry, transforms=(transform_entry,), code=code
    )
    recovered = ExprComposer.from_expr(original.expr, catalog)

    assert recovered.source.name == source_entry.name
    assert len(recovered.transforms) == 1
    assert recovered.transforms[0].name == transform_entry.name
    assert recovered.code == code


def test_from_expr_code_only(catalog_with_entries):
    catalog, source_entry, _ = catalog_with_entries
    code = "source.filter(source.amount > 15)"
    original = ExprComposer(source=source_entry, code=code)
    recovered = ExprComposer.from_expr(original.expr, catalog)

    assert recovered.source.name == source_entry.name
    assert recovered.transforms == ()
    assert recovered.code == code


def test_from_expr_with_alias(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    original = ExprComposer(
        source=source_entry, transforms=(transform_entry,), alias="custom-alias"
    )
    recovered = ExprComposer.from_expr(original.expr, catalog)

    assert recovered.alias == "custom-alias"


def test_from_expr_chained_transforms(catalog_with_entries):
    catalog, source_entry, transform_entry = catalog_with_entries
    output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15))

    original = ExprComposer(source=source_entry, transforms=(transform_entry, t2_entry))
    recovered = ExprComposer.from_expr(original.expr, catalog)

    assert recovered.source.name == source_entry.name
    assert len(recovered.transforms) == 2
    assert recovered.transforms[0].name == transform_entry.name
    assert recovered.transforms[1].name == t2_entry.name


def test_from_expr_no_tags_raises(catalog_with_entries):
    catalog, _, _ = catalog_with_entries
    bare = xo.memtable({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="No catalog-source tag found"):
        ExprComposer.from_expr(bare, catalog)


# --- fuse_catalog_source tests ---


def test_fuse_strips_catalog_wrappers(catalog_with_entries):
    """Fusing a bound expression removes all CatalogTag HashingTag nodes."""
    _, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)

    tags_before = walk_nodes(HashingTag, bound)
    catalog_tags_before = tuple(
        t for t in tags_before if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    assert len(catalog_tags_before) >= 2  # at least SOURCE + TRANSFORM

    fused = fuse_catalog_source(bound)
    tags_after = walk_nodes(HashingTag, fused)
    catalog_tags_after = tuple(
        t for t in (tags_after or ()) if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    assert len(catalog_tags_after) == 0


def test_fuse_strips_catalog_remote_tables(catalog_with_entries):
    """Fusing removes all RemoteTable wrappers created by bind."""
    _, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    fused = fuse_catalog_source(bound)

    rts = walk_nodes(RemoteTable, fused)
    assert not rts


def test_fuse_preserves_correctness(catalog_with_entries):
    """Fused expression produces the same result as the unfused one."""
    _, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    fused = fuse_catalog_source(bound)

    expected = bound.execute()
    actual = fused.execute()
    assert actual.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_fuse_chained_transforms(catalog_with_entries):
    """Fusing a multi-transform chain strips all intermediate wrappers."""
    catalog, source_entry, transform_entry = catalog_with_entries
    output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
    ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
    t2_entry = catalog.add(ub2.filter(ub2.amount > 15))

    bound = bind(source_entry, transform_entry, t2_entry)
    fused = fuse_catalog_source(bound)

    tags_after = walk_nodes(HashingTag, fused)
    catalog_tags_after = tuple(
        t for t in (tags_after or ()) if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    assert len(catalog_tags_after) == 0

    expected = bound.execute()
    actual = fused.execute()
    assert actual.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_fuse_bare_source(catalog_with_entries):
    """Fusing a bare source (no transforms) strips its catalog wrappers."""
    _, source_entry, _ = catalog_with_entries
    source_expr = _make_source_expr(source_entry)

    fused = fuse_catalog_source(source_expr)
    tags_after = walk_nodes(HashingTag, fused)
    catalog_tags_after = tuple(
        t for t in (tags_after or ()) if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    assert len(catalog_tags_after) == 0


def test_fuse_noop_without_catalog_tags():
    """Fusing an expression without CatalogTag markers returns it unchanged."""
    plain = xo.memtable({"x": [1, 2, 3]})
    result = fuse_catalog_source(plain)
    assert result is plain


def test_fuse_idempotent(catalog_with_entries):
    """Fusing an already-fused expression returns it unchanged."""
    _, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)
    fused = fuse_catalog_source(bound)
    fused_again = fuse_catalog_source(fused)
    assert fused_again is fused


def test_fuse_read_source(catalog, tmpdir):
    """Fusing strips catalog wrappers even when the source contains Read ops."""
    pq_path = Path(tmpdir) / "data.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3], "y": [4, 5, 6]}), pq_path)

    source_expr = deferred_read_parquet(pq_path)
    reads = walk_nodes(Read, source_expr)
    assert reads, "Expected Read ops for a deferred parquet source"

    source_entry = catalog.add(source_expr)
    transform_schema = source_expr.schema()
    ub = ops.UnboundTable(name="ph", schema=transform_schema).to_expr()
    transform_entry = catalog.add(ub.limit(1))
    bound = bind(source_entry, transform_entry)

    result = fuse_catalog_source(bound)
    # Catalog wrappers should be stripped
    tags = walk_nodes(HashingTag, result)
    catalog_tags = tuple(
        t for t in (tags or ()) if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    assert len(catalog_tags) == 0

    # Read nodes should still be present
    assert walk_nodes(Read, result)

    # Results should be correct
    expected = bound.execute()
    actual = result.execute()
    assert actual.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_fuse_preserves_non_catalog_hashing_tag(catalog_with_entries):
    """Fusing strips CatalogTag wrappers but preserves unrelated HashingTags."""
    _, source_entry, transform_entry = catalog_with_entries
    bound = bind(source_entry, transform_entry)

    # Add a non-catalog HashingTag on top
    tagged = bound.hashing_tag("custom-provenance", author="test")
    fused = fuse_catalog_source(tagged)

    tags = walk_nodes(HashingTag, fused) or ()
    catalog_tags = tuple(
        t for t in tags if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    custom_tags = tuple(t for t in tags if t.metadata.get("tag") == "custom-provenance")
    assert len(catalog_tags) == 0
    assert len(custom_tags) == 1
    assert custom_tags[0].metadata["author"] == "test"


def test_fuse_preserves_non_catalog_remote_table(source_expr):
    """Fusing does not strip user-created RemoteTables (e.g. into_backend)."""
    con = xo.connect()
    remote = source_expr.into_backend(con, name="user_rt")

    # No catalog tags → fuse is a no-op, RemoteTable preserved
    result = fuse_catalog_source(remote)
    assert result is remote

    rts = walk_nodes(RemoteTable, result)
    assert len(rts) == 1


def test_fuse_preserves_non_catalog_remote_table_inside_bind(catalog):
    """User-created RemoteTables inside a source expression survive fusing."""
    con = xo.connect()
    inner = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    remote_source = inner.into_backend(con, name="user_rt")
    source_entry = catalog.add(remote_source, aliases=("remote-src",))

    schema = remote_source.schema()
    ub = ops.UnboundTable(name="ph", schema=schema).to_expr()
    transform_entry = catalog.add(ub.filter(ub.amount > 0).select("user_id", "amount"))

    bound = bind(source_entry, transform_entry)
    fused = fuse_catalog_source(bound)

    # Catalog wrappers stripped
    tags = walk_nodes(HashingTag, fused) or ()
    catalog_tags = tuple(
        t for t in tags if t.metadata.get("tag") in frozenset(CatalogTag)
    )
    assert len(catalog_tags) == 0

    # The user-created RemoteTable is still present
    rts = walk_nodes(RemoteTable, fused)
    assert len(rts) == 1

    # Still produces correct results
    expected = bound.execute()
    actual = fused.execute()
    assert actual.reset_index(drop=True).equals(expected.reset_index(drop=True))
