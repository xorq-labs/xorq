import pandas as pd
import pytest

import xorq.api as xo


@pytest.mark.parametrize(
    "labels",
    [pytest.param(["b", "a"], id="agree"), pytest.param(["a", "b"], id="conflict")],
)
def test_order_by_then_select_alias_shadowing_sort_column(labels):
    # https://github.com/xorq-labs/xorq/issues/2407, on the vendored ibis compiler
    df = pd.DataFrame({"cohort_month": [2, 1], "cohort_label": labels})
    t = xo.datafusion.connect().create_table("t", df)
    expr = t.order_by("cohort_month").select(cohort_month=t.cohort_label)
    assert expr.execute()["cohort_month"].tolist() == labels[::-1]
