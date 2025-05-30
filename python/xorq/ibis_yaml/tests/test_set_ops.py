import random

import pytest
from pytest import param

import xorq as xo
from xorq import _


@pytest.fixture
def union_subsets(alltypes):
    randomizer = random.Random(42)
    cols_a, cols_b, cols_c = (
        randomizer.sample(alltypes.columns, k=len(alltypes.columns)) for _ in range(3)
    )
    assert cols_a != cols_b != cols_c

    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))[cols_a]
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))[cols_b]
    c = alltypes.filter((_.id >= 5213) & (_.id <= 5220))[cols_c]

    return a, b, c


@pytest.mark.parametrize("distinct", [False, True], ids=["all", "distinct"])
def test_union(compiler, union_subsets, distinct):
    a, b, c = union_subsets

    expr = xo.union(a, b, distinct=distinct).order_by("id")
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert roundtrip_expr.equals(expr)


def test_union_mixed_distinct(compiler, union_subsets):
    a, b, c = union_subsets

    expr = a.union(b, distinct=True).union(c, distinct=False).order_by("id")
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert roundtrip_expr.equals(expr)


@pytest.mark.parametrize(
    "distinct",
    [
        param(False, id="all"),
        param(True, id="distinct"),
    ],
)
def test_intersect(compiler, alltypes, distinct):
    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))
    c = alltypes.filter((_.id >= 5195) & (_.id <= 5208))

    expr = xo.intersect(a, b, c, distinct=distinct).order_by("id")
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert roundtrip_expr.equals(expr)


@pytest.mark.parametrize(
    "distinct",
    [
        param(False, id="all"),
        param(True, id="distinct"),
    ],
)
def test_difference(compiler, alltypes, distinct):
    a = alltypes.filter((_.id >= 5200) & (_.id <= 5210))
    b = alltypes.filter((_.id >= 5205) & (_.id <= 5215))
    c = alltypes.filter((_.id >= 5195) & (_.id <= 5202))

    expr = xo.difference(a, b, c, distinct=distinct).order_by("id")
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert roundtrip_expr.equals(expr)
