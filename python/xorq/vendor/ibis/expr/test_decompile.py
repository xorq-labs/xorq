"""Tests for the expression decompiler."""

import pytest

import xorq.vendor.ibis as ibis
from xorq.vendor.ibis.expr.decompile import decompile


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
