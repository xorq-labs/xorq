"""XGBoost model loaded from JSON with expression rewriting for optimization.

Traditional approach: Load an XGBoost model from a JSON file, call predict in
Python per batch. The prediction step is opaque to any query planner, so no
query-level optimization is possible -- all columns are always fetched
regardless of what the model actually needs.

With xorq: Wrap the model in a UDF using make_quickgrove_udf and apply
rewrite_quickgrove_expr to enable expression rewriting. The query optimizer
can then prune unused columns and push operations closer to the data source,
reducing data transfer and improving performance.
"""

from pathlib import Path

import pandas as pd

import xorq.api as xo
from xorq.expr.relations import into_backend
from xorq.ml import make_quickgrove_udf, rewrite_quickgrove_expr


model_path = Path(xo.options.pins.get_path("diamonds-model"))

# source backend
pg = xo.postgres.connect_examples()
# local backend
con = xo.connect()
# load xgboost json model
# cannot have hyphen in the model name see #498
model = make_quickgrove_udf(model_path, model_name="diamonds_model")

t = (
    into_backend(pg.tables["diamonds"], con, "diamonds")
    .mutate(pred=model.on_expr)
    .filter(xo._.carat < 1)
    .select(xo._.pred)
)
t_pruned = rewrite_quickgrove_expr(t)


if __name__ == "__pytest_main__":
    original = xo.execute(t)
    pruned = xo.execute(t_pruned)
    pd.testing.assert_frame_equal(original, pruned, rtol=3)
    pytest_examples_passed = True
