import numpy as np
import pandas as pd

import xorq as xo
import xorq.expr.datatypes as dt
from xorq import udf


array_types_df = pd.DataFrame(
    [
        (
            [np.int64(1), 2, 3],
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            "a",
            1.0,
            [[], [np.int64(1), 2, 3], None],
        ),
        (
            [4, 5],
            ["d", "e"],
            [4.0, 5.0],
            "a",
            2.0,
            [],
        ),
        (
            [6, None],
            ["f", None],
            [6.0, np.nan],
            "a",
            3.0,
            [None, [], None],
        ),
        (
            [None, 1, None],
            [None, "a", None],
            [],
            "b",
            4.0,
            [[1], [2], [], [3, 4, 5]],
        ),
        (
            [2, None, 3],
            ["b", None, "c"],
            np.nan,
            "b",
            5.0,
            None,
        ),
        (
            [4, None, None, 5],
            ["d", None, None, "e"],
            [4.0, np.nan, np.nan, 5.0],
            "c",
            6.0,
            [[1, 2, 3]],
        ),
    ],
    columns=[
        "x",
        "y",
        "z",
        "grouper",
        "scalar_column",
        "multi_dim",
    ],
)

con = xo.connect()
expr = con.register(array_types_df).select("x")
schema = expr.schema()


@udf.agg.pandas_df(schema=schema, return_type=dt.float64, name="explode_sum")
def explode_sum(df):
    return df["x"].explode().dropna().astype(float).sum()


expr = expr.tag("full")
expr = explode_sum.on_expr(expr).as_table()

if __name__ == "__pytest_main__":
    assert expr.execute() is not None
