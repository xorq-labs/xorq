import ibis
import ibis.expr.operations as ops

from xorq.common.utils.ibis_utils import from_ibis


def test_scalar_subquery(t):
    expr = ops.ScalarSubquery(t.c.mean().as_table()).to_expr()

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_exists_subquery():
    t1 = ibis.table(dict(a="int", b="string"), name="t1")
    t2 = ibis.table(dict(a="int", c="float"), name="t2")

    filtered = t2.filter(t2.a == t1.a)
    expr = ops.ExistsSubquery(filtered).to_expr()

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_in_subquery():
    t1 = ibis.table(dict(a="int", b="string"), name="t1")
    t2 = ibis.table(dict(a="int", c="float"), name="t2")

    expr = ops.InSubquery(t1.select("a"), t2.a).to_expr()

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
