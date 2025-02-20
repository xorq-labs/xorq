from __future__ import annotations

from xorq.tests.util import assert_frame_equal


def test_where_multiple_conditions(alltypes, df):
    expr = alltypes.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    result = expr.execute()

    expected = df[
        (df["float_col"] > 0)
        & (df["smallint_col"] == 9)
        & (df["int_col"] < df["float_col"] * 2)
    ]

    assert_frame_equal(result, expected)
