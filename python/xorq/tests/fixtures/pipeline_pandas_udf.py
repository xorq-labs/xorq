import numpy as np
import pandas as pd

import xorq as xo
import xorq.expr.datatypes as dt
from xorq import udf
from xorq.caching import SourceStorage


df = pd.DataFrame(
    [np.int64(1), 2, 3, 4, 5, 6, None, None, 1, None, 2, None, 3, 4, None, None, 5],
    columns=["x"],
)


ddb = xo.duckdb.connect()
expr = ddb.read_in_memory(df).cache(SourceStorage()).select("x")
schema = expr.schema()


@udf.agg.pandas_df(schema=schema, return_type=dt.float64, name="explode_sum")
def explode_sum(df):
    return df["x"].dropna().astype(float).sum()


expr = expr.tag("full")
expr = explode_sum.on_expr(expr).as_table()

if __name__ == "__pytest_main__":
    assert expr.execute() is not None
