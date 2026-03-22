from pathlib import Path

import pytest

import xorq.api as xo
from xorq.catalog.bind import _validate_schema, bind
from xorq.catalog.catalog import Catalog
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.types.core import ExprMetadata


@pytest.fixture
def catalog(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("bind-repo"))
    return Catalog(repo=repo)


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


class TestExprKind:
    def test_source_kind(self, source_expr):
        meta = ExprMetadata(source_expr)
        assert meta.kind == ExprKind.Source

    def test_unbound_kind(self, transform_expr):
        meta = ExprMetadata(transform_expr)
        assert meta.kind == ExprKind.UnboundExpr
        assert meta.schema_in is not None

    def test_bound_kind(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = bind(source_entry, transform_entry)
        meta = ExprMetadata(bound)
        assert meta.kind == ExprKind.Composed
        assert len(meta.sources) == 2
        kinds = {s["kind"] for s in meta.sources}
        assert kinds == {"source", "unbound_expr"}

    def test_bound_to_dict_includes_sources(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = bind(source_entry, transform_entry)
        meta = ExprMetadata(bound)
        d = meta.to_dict()
        assert d["kind"] == "composed"
        assert "sources" in d
        assert len(d["sources"]) == 2

    def test_chained_bind_kind(self, catalog_with_bound):
        catalog, bound_entry, transform2_entry = catalog_with_bound
        bound2 = bind(bound_entry, transform2_entry)
        meta = ExprMetadata(bound2)
        assert meta.kind == ExprKind.Composed
        assert len(meta.sources) >= 3


# --- Schema validation tests ---


class TestSchemaValidation:
    def test_exact_match(self):
        s = Schema({"a": "int64", "b": "string"})
        _validate_schema(s, s, "src", "trn")

    def test_superset_match(self):
        source = Schema({"a": "int64", "b": "string", "c": "float64"})
        transform = Schema({"a": "int64", "b": "string"})
        _validate_schema(source, transform, "src", "trn")

    def test_missing_column(self):
        source = Schema({"a": "int64"})
        transform = Schema({"a": "int64", "b": "string"})
        with pytest.raises(ValueError, match="missing"):
            _validate_schema(source, transform, "src", "trn")

    def test_type_mismatch(self):
        source = Schema({"a": "int64", "b": "string"})
        transform = Schema({"a": "int64", "b": "float64"})
        with pytest.raises(ValueError, match="type mismatch"):
            _validate_schema(source, transform, "src", "trn")

    def test_missing_and_type_mismatch(self):
        source = Schema({"a": "int64", "b": "string"})
        transform = Schema({"a": "float64", "c": "int64"})
        with pytest.raises(ValueError, match="missing") as exc_info:
            _validate_schema(source, transform, "src", "trn")
        assert "type mismatch" in str(exc_info.value)


# --- Bind tests ---


class TestBind:
    def test_bind_produces_expr(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = bind(source_entry, transform_entry)
        assert bound is not None
        meta = ExprMetadata(bound)
        assert meta.kind == ExprKind.Composed

    def test_bind_not_unbound_raises(self, catalog_with_entries):
        catalog, source_entry, _ = catalog_with_entries
        with pytest.raises(ValueError, match="no UnboundTable"):
            bind(source_entry, source_entry)

    def test_bind_roundtrip_catalog(self, catalog_with_entries):
        """Bound entry can be added to catalog and loaded back."""
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = bind(source_entry, transform_entry)
        bound_entry = catalog.add(bound, aliases=("bound-result",))
        assert bound_entry.kind == ExprKind.Composed
        assert len(bound_entry.sources) == 2

    def test_bind_with_alias(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = bind(source_entry, transform_entry, alias="custom-alias")
        meta = ExprMetadata(bound)
        source_sources = tuple(s for s in meta.sources if s["kind"] == "source")
        assert source_sources[0]["alias"] == "custom-alias"

    def test_bind_bound_entry_as_source(self, catalog_with_bound):
        """A bound entry can be used as the source for another bind."""
        catalog, bound_entry, transform2_entry = catalog_with_bound
        bound2 = bind(bound_entry, transform2_entry)
        assert bound2 is not None
        meta = ExprMetadata(bound2)
        assert meta.kind == ExprKind.Composed

    def test_bind_bound_entry_executes(self, catalog_with_bound):
        """Binding a bound entry produces an executable expression."""
        catalog, bound_entry, transform2_entry = catalog_with_bound
        bound2 = bind(bound_entry, transform2_entry)
        result = bound2.execute()
        assert len(result) == 2
        assert set(result["user_id"]) == {2, 3}

    def test_bind_bound_roundtrip_catalog(self, catalog_with_bound):
        """Chained bind can be added to catalog and loaded back."""
        catalog, bound_entry, transform2_entry = catalog_with_bound
        bound2 = bind(bound_entry, transform2_entry)
        bound2_entry = catalog.add(bound2, aliases=("bound2",))
        assert bound2_entry.kind == ExprKind.Composed
        assert len(bound2_entry.sources) >= 2

    def test_bind_schema_mismatch(self, catalog):
        """Binding incompatible schemas raises ValueError."""
        source = xo.memtable({"x": [1, 2], "y": ["a", "b"]})
        schema = xo.Schema({"a": "int64", "b": "float64"})
        unbound = ops.UnboundTable(name="placeholder", schema=schema).to_expr()
        transform = unbound.filter(unbound.a > 0)

        source_entry = catalog.add(source)
        transform_entry = catalog.add(transform)

        with pytest.raises(ValueError, match="mismatch"):
            bind(source_entry, transform_entry)

    def test_bind_variadic(self, catalog_with_entries):
        """bind(source, t1, t2) chains all transforms in one call."""
        catalog, source_entry, transform_entry = catalog_with_entries

        output_schema = xo.Schema({"user_id": "int64", "amount": "float64"})
        ub2 = ops.UnboundTable(name="ph2", schema=output_schema).to_expr()
        t2_entry = catalog.add(ub2.filter(ub2.amount > 15), aliases=("t2",))

        bound = bind(source_entry, transform_entry, t2_entry)
        result = bound.execute()
        assert len(result) == 2
        assert set(result["user_id"]) == {2, 3}

    def test_bind_no_transforms_raises(self, catalog_with_entries):
        """bind() with zero transforms raises ValueError."""
        _, source_entry, _ = catalog_with_entries
        with pytest.raises(ValueError, match="At least one transform"):
            bind(source_entry)

    def test_bind_plain_expr_as_transform_raises(self, catalog_with_entries):
        """Using a source (no UnboundTable) as transform raises ValueError."""
        catalog, source_entry, _ = catalog_with_entries
        another_source = xo.memtable({"user_id": [4], "amount": [40.0]})
        another_entry = catalog.add(another_source, aliases=("another-source",))
        with pytest.raises(ValueError, match="no UnboundTable"):
            bind(source_entry, another_entry)


# --- get_catalog_entry tests ---


class TestGetCatalogEntry:
    def test_get_by_name(self, catalog_with_entries):
        catalog, source_entry, _ = catalog_with_entries
        resolved = catalog.get_catalog_entry(source_entry.name)
        assert resolved.name == source_entry.name

    def test_get_by_alias(self, catalog_with_entries):
        catalog, source_entry, _ = catalog_with_entries
        resolved = catalog.get_catalog_entry("my-source", maybe_alias=True)
        assert resolved.name == source_entry.name

    def test_get_unknown_raises(self, catalog_with_entries):
        catalog, _, _ = catalog_with_entries
        with pytest.raises(AssertionError, match="not found"):
            catalog.get_catalog_entry("nonexistent", maybe_alias=True)


# --- catalog.source() tests ---


class TestCatalogSource:
    def test_source_returns_catalog_source_expr(self, catalog_with_entries):
        catalog, source_entry, _ = catalog_with_entries
        expr = catalog.source("my-source")
        assert expr is not None
        meta = ExprMetadata(expr)
        assert meta.kind == ExprKind.Composed
        assert len(meta.sources) == 1
        assert meta.sources[0]["kind"] == "source"
        assert meta.sources[0]["entry_name"] == source_entry.name

    def test_source_by_name(self, catalog_with_entries):
        catalog, source_entry, _ = catalog_with_entries
        expr = catalog.source(source_entry.name)
        meta = ExprMetadata(expr)
        assert meta.sources[0]["kind"] == "source"

    def test_source_executes(self, catalog_with_entries):
        catalog, _, _ = catalog_with_entries
        expr = catalog.source("my-source")
        result = expr.execute()
        assert len(result) == 3


# --- catalog.bind() tests ---


class TestCatalogBind:
    def test_bind_produces_both_kinds(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = catalog.bind(source_entry, transform_entry)
        meta = ExprMetadata(bound)
        assert meta.kind == ExprKind.Composed

        kinds = {s["kind"] for s in meta.sources}
        assert "source" in kinds
        assert "unbound_expr" in kinds

    def test_bind_source_provenance(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = catalog.bind(source_entry, transform_entry)
        meta = ExprMetadata(bound)

        source_entries = tuple(s for s in meta.sources if s["kind"] == "source")
        assert len(source_entries) == 1
        assert source_entries[0]["entry_name"] == source_entry.name

    def test_bind_transform_provenance(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = catalog.bind(source_entry, transform_entry)
        meta = ExprMetadata(bound)

        unbound_entries = tuple(s for s in meta.sources if s["kind"] == "unbound_expr")
        assert len(unbound_entries) == 1
        assert unbound_entries[0]["entry_name"] == transform_entry.name

    def test_bind_executes(self, catalog_with_entries):
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = catalog.bind(source_entry, transform_entry)
        result = bound.execute()
        assert len(result) == 3
        assert set(result.columns) == {"user_id", "amount"}

    def test_bind_not_unbound_raises(self, catalog_with_entries):
        catalog, source_entry, _ = catalog_with_entries
        with pytest.raises(ValueError, match="no UnboundTable"):
            catalog.bind(source_entry, source_entry)

    def test_bind_roundtrip_catalog(self, catalog_with_entries):
        """bind() result can be added to catalog and loaded back with kind info."""
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = catalog.bind(source_entry, transform_entry)
        bound_entry = catalog.add(bound, aliases=("bound-result",))
        assert bound_entry.kind == ExprKind.Composed
        assert len(bound_entry.sources) == 2

        kinds = {s["kind"] for s in bound_entry.sources}
        assert "source" in kinds
        assert "unbound_expr" in kinds

    def test_bind_variadic(self, catalog_with_entries):
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


class TestInlineChaining:
    def test_bind_expr_as_source(self, catalog_with_entries):
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

    def test_chain_three_steps(self, catalog_with_entries):
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

    def test_chain_preserves_transform_provenance(self, catalog_with_entries):
        """Each bind step adds an unbound_expr CatalogSource."""
        catalog, source_entry, transform_entry = catalog_with_entries

        schema2 = xo.Schema({"user_id": "int64", "amount": "float64"})
        ub2 = ops.UnboundTable(name="ph2", schema=schema2).to_expr()
        t2_entry = catalog.add(ub2.filter(ub2.amount > 15), aliases=("t2",))

        bound = bind(source_entry, transform_entry, t2_entry)
        meta = ExprMetadata(bound)
        unbound_sources = tuple(s for s in meta.sources if s["kind"] == "unbound_expr")
        assert len(unbound_sources) == 2


# --- Kind YAML roundtrip test ---


class TestKindYAMLRoundtrip:
    def test_kind_survives_yaml_roundtrip(self, catalog_with_entries):
        """kind field is preserved through build/load (YAML serialization)."""
        catalog, source_entry, transform_entry = catalog_with_entries
        bound = catalog.bind(source_entry, transform_entry)

        bound_entry = catalog.add(bound, aliases=("roundtrip-test",))

        loaded_expr = bound_entry.expr
        meta = ExprMetadata(loaded_expr)

        kinds = {s["kind"] for s in meta.sources}
        assert "source" in kinds
        assert "unbound_expr" in kinds
