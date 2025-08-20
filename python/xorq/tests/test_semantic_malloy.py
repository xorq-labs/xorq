import xorq.vendor.ibis as ibis
import pytest

from xorq.expr.translate import convert  # noqa: F401
from xorq.semantic.api import (
    aggregate_,
    group_by_,
    mutate_,
    to_semantic_table,
    with_dimensions,
    with_measures,
)


@pytest.mark.parametrize(
    "data, keys, measure_name, expected_agg",
    [
        (
            {"origin": ["A", "B", "A"], "value": [1, 2, 3]},
            ("origin",),
            "total",
            lambda tbl: tbl.value.sum(),
        ),
    ],
)
def test_semantic_group_by_aggregate(data, keys, measure_name, expected_agg):
    tbl = ibis.memtable(data, name="tbl")
    sem = to_semantic_table(tbl)
    sem = with_dimensions(sem, origin=lambda t: t.origin)
    sem = with_measures(sem, **{measure_name: lambda t: t.value.sum()})

    q = aggregate_(group_by_(sem, *keys), **{measure_name: lambda t: t.value.sum()})
    expr = convert(q, catalog={})
    expected = tbl.group_by([tbl.origin]).aggregate(**{measure_name: expected_agg(tbl)})
    assert repr(expr) == repr(expected)


def test_semantic_post_agg_mutate():
    data = {"origin": ["A", "A", "B"], "value": [10, 20, 5]}
    tbl = ibis.memtable(data, name="tbl")

    sem = to_semantic_table(tbl)
    sem = with_dimensions(sem, origin=lambda t: t.origin)
    sem = with_measures(sem, count=lambda t: t.value.count())

    gb = group_by_(sem, "origin")
    agg = aggregate_(gb, count=lambda t: t.value.count())
    q = mutate_(agg, new=lambda t: t.count + 1)

    expr = convert(q, catalog={})
    expected = (
        tbl.group_by([tbl.origin])
        .aggregate(count=tbl.value.count())
        .mutate(new=(tbl.value.count() + 1))
    )
    assert repr(expr) == repr(expected)
