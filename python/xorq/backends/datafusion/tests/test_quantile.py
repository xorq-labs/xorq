import pytest

import xorq.api as xo


@pytest.fixture
def t():
    con = xo.datafusion.connect()
    return con.create_table(
        "t",
        {
            "a": [3.0, None, 1.0, 4.0, 2.0, 10.0],
            "i": [3, None, 1, 4, 2, 10],
            "s": ["c", None, "a", "d", "b", "e"],
            "inf": [float("inf")] * 6,
            "big": [-1e308, None, 1e308, 1e308, 1e308, 1e308],
            "g": [1, 1, 1, 2, 2, 2],
        },
    )


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(lambda t: t.a.quantile(0.5), 3.0, id="median"),
        pytest.param(lambda t: t.a.quantile(0), 1.0, id="min"),
        pytest.param(lambda t: t.a.quantile(1), 10.0, id="max"),
        pytest.param(lambda t: t.i.quantile(0.9), 7.6, id="interpolated_int"),
        pytest.param(lambda t: t.a.quantile(0.5, where=t.g == 2), 4.0, id="where"),
        pytest.param(lambda t: t.inf.quantile(0.5), float("inf"), id="constant_inf"),
        pytest.param(lambda t: t.big.quantile(0.125), 0.0, id="no_overflow"),
    ],
)
def test_quantile(t, make_expr, expected):
    assert make_expr(t).execute() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0, "a"), (0.2, "a"), (0.25, "b"), (0.5, "c"), (0.75, "d"), (0.99, "e"), (1, "e")],
)
def test_quantile_string_is_discrete(t, q, expected):
    assert t.s.quantile(q).execute() == expected


def test_quantile_grouped(t):
    expr = t.group_by("g").agg(q=t.a.quantile(0.5)).order_by("g")
    assert expr.execute().q.tolist() == [2.0, 4.0]
