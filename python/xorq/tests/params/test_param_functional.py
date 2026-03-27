"""Functional test suite for named parameters.

Tests the complete workflow without implementation details.
"""

import datetime

import pytest

import xorq.api as xo
from xorq.expr.operations import NamedScalarParameter
from xorq.ibis_yaml.compiler import build_expr, load_expr


def test_param_creation_basic():
    threshold = xo.param("threshold", "float64")
    assert threshold.op().label == "threshold"
    assert str(threshold.op().dtype) == "float64"


def test_param_with_default():
    limit = xo.param("limit", "int64", default=10)
    assert limit.op().label == "limit"
    assert limit.op().default == 10


def test_param_required():
    cutoff = xo.param("cutoff", "date")
    assert cutoff.op().default is None


def test_param_in_filter():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    result = expr.execute(params={threshold: 1.5})
    assert list(result["x"]) == [2.0, 3.0]


def test_param_multiple_operations():
    lo = xo.param("lo", "float64", default=0.0)
    hi = xo.param("hi", "float64", default=10.0)
    t = xo.memtable({"x": [1.0, 5.0, 10.0, 15.0]})
    expr = t.filter((t.x > lo) & (t.x < hi))

    assert list(expr.execute()["x"]) == [1.0, 5.0]
    assert list(expr.execute(params={lo: 4.0, hi: 12.0})["x"]) == [5.0, 10.0]


def test_param_different_values():
    cutoff = xo.param("cutoff", "date")
    t = xo.memtable(
        {
            "d": [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 6, 1),
                datetime.date(2024, 12, 1),
            ]
        }
    )
    expr = t.filter(t.d >= cutoff)
    assert len(expr.execute(params={cutoff: datetime.date(2024, 6, 1)})) == 2
    assert len(expr.execute(params={cutoff: datetime.date(2024, 12, 1)})) == 1


@pytest.mark.parametrize(
    "dtype, value, data, predicate_fn, expected_len",
    [
        ("float64", 2.0, {"x": [1.1, 2.2, 3.3]}, lambda t, p: t.x > p, 2),
        ("int64", 3, {"x": [1, 2, 3, 4, 5]}, lambda t, p: t.x <= p, 3),
        (
            "date",
            datetime.date(2024, 6, 1),
            {
                "d": [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 6, 1),
                    datetime.date(2024, 12, 1),
                ]
            },
            lambda t, p: t.d >= p,
            2,
        ),
        (
            "timestamp",
            datetime.datetime(2024, 1, 1, 12, 0),
            {
                "ts": [
                    datetime.datetime(2024, 1, 1, 10, 0),
                    datetime.datetime(2024, 1, 1, 12, 0),
                    datetime.datetime(2024, 1, 1, 14, 0),
                ]
            },
            lambda t, p: t.ts >= p,
            2,
        ),
        (
            "string",
            "east",
            {"region": ["north", "south", "east", "west"]},
            lambda t, p: t.region == p,
            1,
        ),
        (
            "boolean",
            True,
            {"active": [True, False, True, False]},
            lambda t, p: t.active == p,
            2,
        ),
    ],
    ids=["float64", "int64", "date", "timestamp", "string", "boolean"],
)
def test_param_type_variants(dtype, value, data, predicate_fn, expected_len):
    p = xo.param("p", dtype)
    t = xo.memtable(data)
    assert len(t.filter(predicate_fn(t, p)).execute(params={p: value})) == expected_len


def test_yaml_round_trip(tmp_path):
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0]})
    expr = t.filter(t.x > threshold)

    build_dir = build_expr(expr, builds_dir=tmp_path)
    restored = load_expr(build_dir)

    params = restored.op().find(NamedScalarParameter)
    assert len(params) == 1
    assert params[0].label == "threshold"
    assert params[0].default == 1.5
    assert list(restored.execute()["x"]) == [2.0]


def test_yaml_multiple_params(tmp_path):
    lo = xo.param("lo", "float64", default=0.0)
    hi = xo.param("hi", "float64", default=10.0)
    t = xo.memtable({"x": [1.0, 5.0, 10.0]})
    expr = t.filter((t.x > lo) & (t.x < hi))

    build_dir = build_expr(expr, builds_dir=tmp_path)
    restored = load_expr(build_dir)

    assert {p.label for p in restored.op().find(NamedScalarParameter)} == {"lo", "hi"}


def test_bind_mixed_params():
    required = xo.param("required", "float64")
    optional = xo.param("optional", "float64", default=11.0)
    t = xo.memtable({"x": [1.0, 5.0, 10.0, 15.0]})
    expr = t.filter((t.x > required) & (t.x < optional))
    assert list(expr.execute(params={required: 5.0})["x"]) == [10.0]


def test_multiple_params_same_expr():
    p1 = xo.param("p1", "float64", default=1.0)
    _ = xo.param("p2", "float64", default=2.0)
    p3 = xo.param("p3", "float64", default=3.0)
    t = xo.memtable({"x": [0.5, 1.5, 2.5, 3.5]})
    assert list(t.filter((t.x > p1) & (t.x < p3)).execute()["x"]) == [1.5, 2.5]


def test_param_in_complex_expr():
    threshold = xo.param("threshold", "float64", default=5.0)
    t = xo.memtable({"x": [1.0, 3.0, 5.0, 7.0, 9.0]})
    expr = (
        t.filter(t.x > threshold)
        .group_by((t.x // 2).name("bucket"))
        .aggregate(count=t.x.count())
    )
    assert len(expr.execute()) == 2


def test_param_with_tagging():
    threshold = xo.param("threshold", "float64", default=0.5)
    t = xo.memtable({"x": [1.0, 2.0]})
    result = t.filter(t.x > threshold).tag("test_param_tag").execute()
    assert len(result) > 0
