import numpy as np
import pandas as pd
from toolz import identity

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq import udf
from xorq.common.utils.toolz_utils import curry
from xorq.vendor import ibis


df = pd.DataFrame(
    [np.int64(1), 2, 3, 4, 5, 6, None, None, 1, None, 2, None, 3, 4, None, None, 5],
    columns=["x"],
)

con = xo.connect()
data = con.register(df, table_name="data").select("x")
schema = data.schema()


@udf.agg.pandas_df(schema=schema, return_type=dt.float64, name="explode_sum")
def explode_sum(df):
    return df["x"].dropna().astype(float).sum()


expr = data.tag("full")
expr = explode_sum.on_expr(expr).name("explode_sum").as_table()


@curry
def add_value(value, frame, **kwargs):
    return (frame["x"] + value).astype(float)


add_sum = udf.make_pandas_expr_udf(
    computed_kwargs_expr=expr,
    fn=add_value,
    schema=ibis.schema({"x": dt.float64}),
    name="add_explode_sum",
    return_type=dt.float64,
    post_process_fn=identity,
)

expr = data.mutate(out=add_sum.on_expr(data)).as_table()


if __name__ == "__pytest_main__":
    assert (res := expr.execute()) is not None
    print(res)
