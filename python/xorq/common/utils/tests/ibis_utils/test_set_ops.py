import random

import pytest
from pytest import param

from xorq.common.utils.ibis_utils import from_ibis


ibis = pytest.importorskip("ibis")


@pytest.fixture
def union_subsets(ibis_alltypes):
    randomizer = random.Random(42)
    cols_a, cols_b, cols_c = (
        randomizer.sample(ibis_alltypes.columns, k=len(ibis_alltypes.columns))
        for _ in range(3)
    )
    assert cols_a != cols_b != cols_c

    a = ibis_alltypes.filter((ibis._.id >= 5200) & (ibis._.id <= 5210))[cols_a]
    b = ibis_alltypes.filter((ibis._.id >= 5205) & (ibis._.id <= 5215))[cols_b]
    c = ibis_alltypes.filter((ibis._.id >= 5213) & (ibis._.id <= 5220))[cols_c]

    return a, b, c


@pytest.mark.parametrize("distinct", [False, True], ids=["all", "distinct"])
def test_union(union_subsets, distinct):
    a, b, c = union_subsets

    expr = ibis.union(a, b, distinct=distinct).order_by("id")

    xorq_expr = from_ibis(expr)
    assert xorq_expr.execute() is not None


def test_union_mixed_distinct(union_subsets):
    a, b, c = union_subsets

    expr = a.union(b, distinct=True).union(c, distinct=False).order_by("id")

    xorq_expr = from_ibis(expr)
    assert xorq_expr.execute() is not None


@pytest.mark.parametrize(
    "distinct",
    [
        param(False, id="all"),
        param(True, id="distinct"),
    ],
)
def test_intersect(ibis_alltypes, distinct):
    a = ibis_alltypes.filter((ibis._.id >= 5200) & (ibis._.id <= 5210))
    b = ibis_alltypes.filter((ibis._.id >= 5205) & (ibis._.id <= 5215))
    c = ibis_alltypes.filter((ibis._.id >= 5195) & (ibis._.id <= 5208))

    expr = ibis.intersect(a, b, c, distinct=distinct).order_by("id")

    xorq_expr = from_ibis(expr)
    assert xorq_expr.execute() is not None


@pytest.mark.parametrize(
    "distinct",
    [
        param(False, id="all"),
        param(True, id="distinct"),
    ],
)
def test_difference(ibis_alltypes, distinct):
    a = ibis_alltypes.filter((ibis._.id >= 5200) & (ibis._.id <= 5210))
    b = ibis_alltypes.filter((ibis._.id >= 5205) & (ibis._.id <= 5215))
    c = ibis_alltypes.filter((ibis._.id >= 5195) & (ibis._.id <= 5202))

    expr = ibis.difference(a, b, c, distinct=distinct).order_by("id")

    xorq_expr = from_ibis(expr)
    assert xorq_expr.execute() is not None
