from pathlib import Path

import pandas as pd

import letsql as ls
from letsql.expr.ml import make_quickgrove_udf, rewrite_quickgrove_expr
from letsql.expr.relations import into_backend


model_path = Path(ls.options.pins.get_path("diamonds-model"))

# source backend
pg = ls.postgres.connect_examples()
# local backend
con = ls.connect()
# load xgboost json model
# cannot have hyphen in the model name see #498
model = make_quickgrove_udf(model_path, model_name="diamonds_model")

t = (
    into_backend(pg.tables["diamonds"], con, "diamonds")
    .mutate(pred=model.on_expr)
    .filter(ls._.carat < 1)
    .select(ls._.pred)
)

t_pruned = rewrite_quickgrove_expr(t)

original = ls.execute(t)
pruned = ls.execute(t_pruned)

pd.testing.assert_frame_equal(original, pruned, rtol=3)
