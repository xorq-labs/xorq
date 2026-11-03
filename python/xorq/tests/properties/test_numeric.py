import functools
import math

import pandas as pd
import pytest
from hypothesis import example, given

import xorq.api as xo
from xorq.tests.strategies import numeric_frames
from xorq.vendor.ibis.util import gen_name


NUMERIC_OPS = {
    "round": lambda t: t.a.round(),
    "round_2": lambda t: t.a.round(2),
    "round_-1": lambda t: t.a.round(-1),
    "between_columns": lambda t: t.a.between(t.b, t.c),
    "between_scalars": lambda t: t.a.between(-1.5, 2.5),
    "add": lambda t: t.a + t.b,
    "sub": lambda t: t.a - t.b,
    "mul": lambda t: t.a * t.b,
}


@functools.cache
def oracle():
    # one connection per process: a function-scoped fixture under @given is a
    # Hypothesis health-check error
    return xo.duckdb.connect()


def results(t, op):
    values = t.select("k", r=NUMERIC_OPS[op](t)).order_by("k").execute().r
    return [None if pd.isna(value) else value for value in values]


def same(left, right):
    match left, right:
        case (None, None):
            return True
        case (None, _) | (_, None):
            return False
        case _:
            return math.isclose(left, right, rel_tol=1e-9)


@pytest.mark.parametrize("op", sorted(NUMERIC_OPS))
@given(frame=numeric_frames())
# round ties go half away from zero, as on duckdb (#2459)
@example(
    frame=pd.DataFrame(
        {"k": range(6)} | dict.fromkeys("abc", [2.5, 3.5, -2.5, -0.5, 0.5, -1.5])
    ),
)
# a nullable Float64 column with NA keeps it through round() (#2459)
@example(
    frame=pd.DataFrame(
        {"k": range(4)}
        | dict.fromkeys("abc", pd.Series([1.26, 2.5, None, 4.75], dtype="Float64"))
    ),
)
# a null value or bound makes between null, not False (#2460)
@example(
    frame=pd.DataFrame(
        {
            "k": range(7),
            "a": [1.26, 2.5, None, 4.75, 2.0, 5.0, 5.0],
            "b": [0, 3, 0, 4, None, 10, None],
            "c": [2, 3, 5, None, 5, None, 1],
        }
    ),
)
def test_pandas_numeric_matches_duckdb(op, frame):
    name = gen_name("numeric")
    table = oracle().create_table(name, frame)
    try:
        expected = results(table, op)
    finally:
        oracle().drop_table(name)
    actual = results(xo.pandas.connect({"t": frame}).table("t"), op)

    assert len(actual) == len(expected)
    assert all(map(same, actual, expected)), (actual, expected)
