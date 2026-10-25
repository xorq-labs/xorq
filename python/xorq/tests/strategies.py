"""Hypothesis strategies for property tests that compare a backend with an oracle.

The numeric family holds the values duckdb (the oracle) and the backends under
test agree on: no infinity, and magnitudes far below 2**53, so a disagreement
is a bug in the backend rather than in float precision. Ties are drawn on
purpose: halves for round(), odd multiples of 5 for round(-1).
"""

import hypothesis.strategies as st
import pandas as pd


# |x| <= 1e6, so a product of two stays far below 2**53
FLOATS = st.one_of(
    st.integers(-200, 200).map(lambda n: n / 2),
    st.integers(-200, 200).map(lambda n: n * 5.0),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
INTEGERS = st.integers(-(10**6), 10**6)

# dtype -> value strategy; only the nullable dtypes (and float64, whose NaN
# arrives as null) can hold a missing value
COLUMN_VALUES = {
    "float64": st.one_of(FLOATS, st.none()),
    "Float64": st.one_of(FLOATS, st.none()),
    "int64": INTEGERS,
    "Int64": st.one_of(INTEGERS, st.none()),
}


@st.composite
def numeric_frames(draw, max_rows=20):
    """Frames with a unique int64 key ``k`` and numeric columns a, b, c, each
    with its own dtype, so mixed float/int and numpy/nullable pairs come up."""
    n = draw(st.integers(0, max_rows))

    def column():
        dtype = draw(st.sampled_from(sorted(COLUMN_VALUES)))
        values = draw(st.lists(COLUMN_VALUES[dtype], min_size=n, max_size=n))
        return pd.Series(values, dtype=dtype)

    return pd.DataFrame({"k": range(n), **{name: column() for name in "abc"}})


# -- relational shapes ----------------------------------------------------------
#
# A shape is a nested tuple naming a relational step and its inputs, with
# ("base",) for the table under test. Every step keeps exactly the columns
# k (int64) and v (float64), so any step composes with any other and a union
# of two shapes is always valid. A shape can read the base table once, or
# several times through a union, a self-join or a subquery, which is what the
# scan counts that size a RemoteTable's cache have to get right.

KEYS = st.integers(0, 3)
VALUES = st.one_of(st.integers(-50, 50).map(float), st.none())


@st.composite
def kv_frames(draw, max_rows=12):
    """Frames with a small, repeating int64 key ``k`` (so joins fan out) and a
    nullable float64 ``v``; never empty, since the xorq backend cannot
    create a table from a zero-row frame."""
    rows = draw(st.lists(st.tuples(KEYS, VALUES), min_size=1, max_size=max_rows))
    return pd.DataFrame(
        {
            "k": pd.Series([k for k, _ in rows], dtype="int64"),
            "v": pd.Series([v for _, v in rows], dtype="float64"),
        }
    )


def _unary(child):
    return st.one_of(
        st.tuples(st.just("filter"), child, st.integers(-10, 10)),
        st.tuples(st.just("above_mean"), child),
        st.tuples(st.just("mutate"), child),
        st.tuples(st.just("distinct"), child),
        st.tuples(st.just("window"), child),
        st.tuples(st.just("top"), child, st.integers(0, 5)),
        st.tuples(st.just("join_agg"), child),
        st.tuples(st.just("self_join"), child),
        st.tuples(st.just("twice"), child),
        st.tuples(st.just("isin"), child),
        st.tuples(st.just("semi"), child),
        st.tuples(st.just("minus_mean"), child),
        st.tuples(st.just("lag"), child),
        st.tuples(st.just("intersect"), child),
    )


def _binary(child):
    return st.tuples(st.just("union"), child, child)


shapes = st.recursive(
    st.just(("base",)),
    lambda child: st.one_of(_unary(child), _binary(child)),
    max_leaves=4,
)


def build_shape(shape, base):
    """The expression ``shape`` describes, over the table ``base``."""
    match shape:
        case ("base",):
            return base
        case ("filter", child, threshold):
            t = build_shape(child, base)
            return t.filter(t.v > threshold)
        case ("above_mean", child):
            t = build_shape(child, base)
            return t.filter(t.v > t.v.mean())
        case ("mutate", child):
            t = build_shape(child, base)
            return t.mutate(v=t.v * 2.0 + t.k)
        case ("distinct", child):
            return build_shape(child, base).distinct()
        case ("window", child):
            t = build_shape(child, base)
            return t.mutate(v=t.v.sum().over(group_by="k"))
        case ("top", child, n):
            t = build_shape(child, base)
            return t.order_by(["k", "v"]).limit(n)
        case ("join_agg", child):
            t = build_shape(child, base)
            agg = t.group_by("k").agg(m=t.v.max())
            return t.join(agg, "k").select("k", v=t.v + agg.m)
        case ("self_join", child):
            # joined on both columns against its own distinct rows, so each
            # row matches at most one: two scans without squaring the rows
            t = build_shape(child, base)
            right = t.distinct().view()
            return t.join(right, ["k", "v"]).select("k", v=t.v + right.v)
        case ("twice", child):
            # the same subexpression read by both sides
            t = build_shape(child, base)
            return t.union(t)
        case ("isin", child):
            t = build_shape(child, base)
            return t.filter(t.k.isin(t.filter(t.v > 0).k))
        case ("semi", child):
            t = build_shape(child, base)
            return t.semi_join(t.filter(t.v > 0), "k")
        case ("minus_mean", child):
            # a scalar subquery in a projection (a bare mean() there would
            # become a window over the same scan)
            t = build_shape(child, base)
            return t.mutate(v=t.v - t.v.mean().as_scalar())
        case ("lag", child):
            # an ordered window, which no backend re-scans
            t = build_shape(child, base)
            return t.mutate(v=t.v.lag().over(group_by="k", order_by="v"))
        case ("intersect", child):
            t = build_shape(child, base)
            return t.intersect(t.filter(t.v > 0))
        case ("union", left, right):
            return build_shape(left, base).union(build_shape(right, base))
    raise ValueError(f"unknown shape {shape!r}")
