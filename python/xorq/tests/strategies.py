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
