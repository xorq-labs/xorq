import pytest

import xorq.api as xo
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import ParquetSnapshotCache
from xorq.expr.relations import CachedNode, FlightUDXF, HashingTag, Tag
from xorq.ibis_yaml.enums import ExprKind
from xorq.vendor.ibis.expr.types.core import ExprMetadata


@pytest.mark.parametrize(
    "schema_val,pattern",
    [
        (ibis.schema({"x": "float64"}), "Schema validation failed, expected"),
        (None, "^Schema validation failed$"),
    ],
)
def test_flight_udxf_validate_schema_fail(schema_val, pattern):
    schema = ibis.schema({"x": "int64"})
    input_expr = ibis.table(schema, name="t")

    class DummyUDXF:
        schema_in_required = schema_val

        @staticmethod
        def schema_in_condition(sch):
            return False

        @staticmethod
        def calc_schema_out(sch):
            return None

    with pytest.raises(ValueError, match=pattern):
        FlightUDXF.validate_schema(input_expr, DummyUDXF)


# -- ExprKind.Source detection ------------------------------------------------


def test_kind_memtable_is_source():
    expr = xo.memtable({"a": [1, 2, 3]})
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Source


def test_kind_database_table_is_source():
    con = xo.connect()
    con.raw_sql("CREATE TABLE _test_src (a INT, b VARCHAR)")
    expr = con.table("_test_src")
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Source


def test_kind_deferred_read_is_source():
    expr = xo.deferred_read_parquet(
        "s3://bucket/data.parquet",
        schema={"a": "int64", "b": "string"},
    )
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Source


def test_kind_cached_node_is_source():
    con = xo.connect()
    con.raw_sql("CREATE TABLE _test_cached (a INT)")
    table_expr = con.table("_test_cached")
    dt = table_expr.op()
    cached = CachedNode(
        name="_test_cached",
        schema=dt.schema,
        source=dt.source,
        parent=table_expr,
        cache=ParquetSnapshotCache.from_kwargs(),
    )
    assert ExprMetadata.from_expr(cached.to_expr()).kind == ExprKind.Source


def test_kind_tagged_source_is_source():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = Tag(schema=schema, parent=inner)
    assert ExprMetadata.from_expr(tagged.to_expr()).kind == ExprKind.Source


def test_kind_hashing_tagged_source_is_source():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = HashingTag(schema=schema, parent=inner)
    assert ExprMetadata.from_expr(tagged.to_expr()).kind == ExprKind.Source


def test_kind_filtered_table_is_expr():
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Expr


def test_kind_projected_table_is_expr():
    expr = xo.memtable({"a": [1, 2], "b": [3, 4]}).select("a")
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Expr


def test_kind_aggregated_table_is_expr():
    t = xo.memtable({"a": [1, 2, 3]})
    expr = t.aggregate(total=t.a.sum())
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Expr


def test_kind_unbound_table_is_unbound_expr():
    t = xo.table(schema={"a": "int64"})
    assert ExprMetadata.from_expr(t).kind == ExprKind.UnboundExpr


def test_kind_filtered_unbound_is_unbound_expr():
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    assert ExprMetadata.from_expr(expr).kind == ExprKind.UnboundExpr


def test_kind_source_to_dict():
    expr = xo.memtable({"a": [1, 2, 3]})
    d = ExprMetadata.from_expr(expr).to_dict()
    assert d["kind"] == "source"
    assert "schema_out" in d
    assert "schema_in" not in d


# -- .ls.kind accessor --------------------------------------------------------


def test_ls_kind_source():
    expr = xo.memtable({"a": [1, 2, 3]})
    assert expr.ls.kind == ExprKind.Source


def test_ls_kind_expr():
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    assert expr.ls.kind == ExprKind.Expr


def test_ls_kind_unbound_expr():
    t = xo.table(schema={"a": "int64"})
    assert t.ls.kind == ExprKind.UnboundExpr


def test_ls_kind_matches_metadata():
    exprs = [
        xo.memtable({"a": [1]}),
        xo.memtable({"a": [1]}).filter(xo._.a > 0),
        xo.table(schema={"a": "int64"}),
    ]
    for expr in exprs:
        assert expr.ls.kind == ExprMetadata.from_expr(expr).kind


# -- .ls.unwrapped accessor ---------------------------------------------------


def test_unwrapped_bare_table():
    expr = xo.memtable({"a": [1, 2, 3]})
    assert isinstance(expr.ls.unwrapped, ops.InMemoryTable)


def test_unwrapped_strips_tag():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = Tag(schema=schema, parent=inner)
    assert isinstance(tagged.to_expr().ls.unwrapped, ops.InMemoryTable)


def test_unwrapped_strips_hashing_tag():
    schema = ibis.schema({"a": "int64"})
    inner = xo.memtable({"a": [1, 2, 3]}).op()
    tagged = HashingTag(schema=schema, parent=inner)
    assert isinstance(tagged.to_expr().ls.unwrapped, ops.InMemoryTable)


def test_unwrapped_preserves_cached_node():
    con = xo.connect()
    con.raw_sql("CREATE TABLE _test_unwrapped (a INT)")
    table_expr = con.table("_test_unwrapped")
    dt = table_expr.op()
    cached = CachedNode(
        name="_test_unwrapped",
        schema=dt.schema,
        source=dt.source,
        parent=table_expr,
    )
    assert isinstance(cached.to_expr().ls.unwrapped, CachedNode)


def test_unwrapped_filtered_table():
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    assert not isinstance(expr.ls.unwrapped, ops.InMemoryTable)
