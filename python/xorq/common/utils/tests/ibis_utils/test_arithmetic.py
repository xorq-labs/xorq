import pytest

from xorq.common.utils.ibis_utils import from_ibis


ibis = pytest.importorskip("ibis")


def test_add():
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 + lit2

    xo_expr = from_ibis(expr)
    assert xo_expr is not None


def test_subtract():
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 - lit2

    xo_expr = from_ibis(expr)
    assert xo_expr is not None


def test_multiply():
    lit1 = ibis.literal(5)
    lit2 = ibis.literal(3)
    expr = lit1 * lit2

    xo_expr = from_ibis(expr)
    assert xo_expr is not None


def test_divide():
    lit1 = ibis.literal(6.0)
    lit2 = ibis.literal(2.0)
    expr = lit1 / lit2

    xo_expr = from_ibis(expr)
    assert xo_expr is not None


def test_mixed_arithmetic():
    i = ibis.literal(5)
    f = ibis.literal(2.5)
    expr = i * f

    xo_expr = from_ibis(expr)
    assert xo_expr is not None


def test_complex_arithmetic():
    a = ibis.literal(10)
    b = ibis.literal(5)
    c = ibis.literal(2.0)
    expr = (a + b) * c

    xo_expr = from_ibis(expr)
    assert xo_expr is not None
