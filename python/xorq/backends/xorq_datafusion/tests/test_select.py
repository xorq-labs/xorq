from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.tests.util import assert_frame_equal


def test_where_multiple_conditions(alltypes, alltypes_df):
    expr = alltypes.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    result = expr.execute()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ]

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "connect",
    [xo.connect, xo.sqlite.connect, xo.pandas.connect],
    ids=["xorq_datafusion", "sqlite", "pandas"],
)
def test_order_by_then_select_alias_shadowing_sort_column(connect):
    # https://github.com/xorq-labs/xorq/issues/2407
    df = pd.DataFrame({"cohort_month": [2, 1], "cohort_label": ["b", "a"]})
    t = connect().create_table("t", df)
    expr = t.order_by("cohort_month").select(cohort_month=t.cohort_label)
    assert expr.execute()["cohort_month"].tolist() == ["a", "b"]
