"""Tests for the expression decompiler."""

import pytest

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.decompile import decompile, translate


class TestDecompileCustomXorqRelations:
    """decompile() should handle custom xorq relation types.

    Regression tests for https://github.com/xorq-labs/xorq/issues/1702

    Currently, CachedNode, RemoteTable, FlightUDXF, and Read all fall back
    to the generic DatabaseTable translate handler, which silently discards
    their extra fields (parent, remote_expr, input_expr, etc.) and emits
    a plain ``ibis.table(...)`` call.
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

    def test_decompile_cached_node_preserves_parent(self):
        """Decompiled CachedNode should reference caching, not just be a plain table."""
        from xorq.expr.relations import CachedNode

        t = ibis.table(name="t", schema={"x": "int64", "y": "string"})
        cached = CachedNode(
            name="placeholder_abc",
            schema=t.schema(),
            source=None,
            parent=t,
            cache=None,
        )
        expr = cached.to_expr()
        code = decompile(expr)
        # Should mention caching — a plain ibis.table() means the parent was lost
        assert "cache" in code.lower(), (
            f"Decompiled CachedNode should reference caching, got:\n{code}"
        )

    def test_decompile_remote_table_preserves_remote_expr(self):
        """Decompiled RemoteTable should reference the remote expression."""
        from xorq.expr.relations import RemoteTable

        t = ibis.table(name="t", schema={"x": "int64", "y": "string"})
        remote = RemoteTable(
            name="placeholder_xyz",
            schema=t.schema(),
            source=None,
            remote_expr=t,
        )
        expr = remote.to_expr()
        code = decompile(expr)
        # Should mention into_backend — a plain ibis.table() means the remote_expr was lost
        assert "into_backend" in code, (
            f"Decompiled RemoteTable should reference remote expr, got:\n{code}"
        )
