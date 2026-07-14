"""Tests for the expression decompiler."""

import pytest

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.decompile import decompile, translate


class TestDecompileZeroArgAnalytics:
    """decompile() should handle zero-argument window/analytic functions.

    Regression test for https://github.com/xorq-labs/xorq/issues/1700
    """

    @pytest.fixture
    def table(self):
        return ibis.table(name="t", schema={"x": "int64", "y": "string"})

    def test_row_number(self, table):
        expr = table.mutate(rn=ibis.row_number())
        code = decompile(expr)
        assert "row_number()" in code

    def test_dense_rank(self, table):
        expr = table.mutate(rn=ibis.dense_rank())
        code = decompile(expr)
        assert "dense_rank()" in code

    def test_rank(self, table):
        expr = table.mutate(rn=ibis.rank())
        code = decompile(expr)
        assert "rank()" in code

    def test_percent_rank(self, table):
        expr = table.mutate(rn=ibis.percent_rank())
        code = decompile(expr)
        assert "percent_rank()" in code

    def test_cume_dist(self, table):
        expr = table.mutate(rn=ibis.cume_dist())
        code = decompile(expr)
        assert "cume_dist()" in code

    def test_row_number_with_filter(self, table):
        """Ensure decompile works when row_number appears in a filter."""
        expr = table.filter(ibis.row_number() < 10)
        code = decompile(expr)
        assert "row_number()" in code

    def test_row_number_combined_with_other_ops(self, table):
        """Ensure decompile works with row_number mixed into larger expressions."""
        expr = table.mutate(
            rn=ibis.row_number(),
            x_doubled=table.x * 2,
        )
        code = decompile(expr)
        assert "row_number()" in code
        assert "t.x" in code


class TestDecompileCustomXorqRelations:
    """decompile() should have handlers for custom xorq relation types.

    Regression tests for https://github.com/xorq-labs/xorq/issues/1702

    CachedNode, RemoteTable, FlightUDXF, and Read should not fall back to
    the generic DatabaseTable handler, which discards their extra fields.
    """

    def test_cached_node_has_translate_handler(self):
        from xorq.expr.relations import CachedNode

        handler = translate.dispatch(CachedNode)
        fallback = translate.dispatch(ops.DatabaseTable)
        assert handler is not fallback, (
            "CachedNode falls back to generic DatabaseTable handler"
        )

    def test_remote_table_has_translate_handler(self):
        from xorq.expr.relations import RemoteTable

        handler = translate.dispatch(RemoteTable)
        fallback = translate.dispatch(ops.DatabaseTable)
        assert handler is not fallback, (
            "RemoteTable falls back to generic DatabaseTable handler"
        )

    def test_flight_udxf_has_translate_handler(self):
        from xorq.expr.relations import FlightUDXF

        handler = translate.dispatch(FlightUDXF)
        fallback = translate.dispatch(ops.DatabaseTable)
        assert handler is not fallback, (
            "FlightUDXF falls back to generic DatabaseTable handler"
        )

    def test_read_has_translate_handler(self):
        from xorq.expr.relations import Read

        handler = translate.dispatch(Read)
        fallback = translate.dispatch(ops.DatabaseTable)
        assert handler is not fallback, (
            "Read falls back to generic DatabaseTable handler"
        )

    def test_decompile_cached_node_mentions_cache(self):
        """Decompiled CachedNode should reference caching, not just be a plain table."""
        from xorq.expr.relations import CachedNode

        t = ibis.table(name="src", schema={"x": "int64", "y": "string"})
        node = CachedNode(
            name="placeholder_abc",
            schema=t.schema(),
            source=None,
            parent=t.op(),
            cache=None,
        )
        code = decompile(node.to_expr())
        # Should mention caching — a plain ibis.table() means the parent was lost
        assert "cache" in code.lower(), (
            f"Decompiled CachedNode should reference caching, got:\n{code}"
        )

    def test_decompile_remote_table_mentions_remote(self):
        """Decompiled RemoteTable should reference the remote expression."""
        from xorq.expr.relations import RemoteTable

        t = ibis.table(name="src", schema={"x": "int64", "y": "string"})
        node = RemoteTable(
            name="placeholder_xyz",
            schema=t.schema(),
            source=None,
            remote_expr=t.op(),
        )
        code = decompile(node.to_expr())
        # Should mention remote/into_backend — a plain ibis.table() means remote_expr was lost
        assert "remote" in code.lower() or "into_backend" in code.lower(), (
            f"Decompiled RemoteTable should reference remote, got:\n{code}"
        )

    def test_decompile_read_mentions_read_method(self):
        """Decompiled Read should include the read method and path."""
        from xorq.expr.relations import Read

        node = Read(
            name="src",
            schema=ibis.schema({"x": "int64", "y": "string"}),
            source=None,
            method_name="read_parquet",
            read_kwargs=(("path", "/data/file.parquet"),),
            normalize_method=lambda x: x,
        )
        code = decompile(node.to_expr())
        assert "read_parquet" in code, (
            f"Decompiled Read should mention read_parquet, got:\n{code}"
        )
        assert "/data/file.parquet" in code, (
            f"Decompiled Read should include the file path, got:\n{code}"
        )

    def test_decompile_flight_udxf_mentions_udxf(self):
        """Decompiled FlightUDXF should reference the UDXF transform."""
        from xorq.expr.relations import FlightUDXF

        t = ibis.table(name="src", schema={"x": "int64", "y": "string"})
        node = FlightUDXF(
            name="my_transform",
            schema=t.schema(),
            source=None,
            input_expr=t.op(),
            udxf=type,
            make_server=lambda: None,
            make_connection=lambda: None,
        )
        code = decompile(node.to_expr())
        assert "udxf" in code.lower() or "flight" in code.lower(), (
            f"Decompiled FlightUDXF should reference UDXF, got:\n{code}"
        )
