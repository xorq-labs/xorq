import letsql as ls
from letsql.common.caching import ParquetCacheStorage


model_name = "diamonds-model"
model_path = ls.options.pins.get_path(model_name)


con = ls.connect()
predict_xgb = con.register_xgb_model(model_name, model_path)
cache = ParquetCacheStorage(source=con, path="letsql-tmp")
pg = ls.postgres.connect(
    # FIXME: use dyndns to point examples.letsql.com to the gcp sql host
    host="34.135.241.141",
    user="letsql",
    password="letsql",
    database="letsql",
)
t = pg.table("diamonds").pipe(con.register, "diamonds").cache(storage=cache)


output = t.mutate(prediction=predict_xgb.on_expr).execute()
output
