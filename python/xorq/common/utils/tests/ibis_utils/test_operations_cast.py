import ibis

from xorq.common.utils.ibis_utils import from_ibis


def test_explicit_cast():
    expr = ibis.literal(42).cast("float64")

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_implicit_cast():
    i = ibis.literal(1)
    f = ibis.literal(2.5)
    expr = i + f

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_string_cast():
    expr = ibis.literal("42").cast("int64")

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_timestamp_cast():
    expr = ibis.literal("2024-01-01").cast("timestamp")

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
