import ibis

from xorq.common.utils.ibis_utils import from_ibis


def test_equals():
    a = ibis.literal(5)
    b = ibis.literal(5)
    expr = a == b

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_not_equals():
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a != b

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_greater_than():
    a = ibis.literal(5)
    b = ibis.literal(3)
    expr = a > b

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_less_than():
    a = ibis.literal(3)
    b = ibis.literal(5)
    expr = a < b

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_and_or():
    a = ibis.literal(5)
    b = ibis.literal(3)
    c = ibis.literal(10)

    expr_and = (a > b) & (a < c)

    xorq_expr = from_ibis(expr_and)
    assert xorq_expr is not None


def test_not():
    a = ibis.literal(True)
    expr = ~a

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_is_null():
    a = ibis.literal(None)
    expr = a.isnull()

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_is_inf():
    a = ibis.literal(float("inf"))
    expr = a.isinf()

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_is_nan():
    from math import isnan

    a = ibis.literal(float("nan"))
    expr = a.isnan()

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert isnan(a.op().value)


def test_between():
    a = ibis.literal(5)
    expr = a.between(3, 7)

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
