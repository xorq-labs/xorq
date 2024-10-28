import tempfile

import letsql as ls
from letsql.common.caching import ParquetCacheStorage


model_name = "diamonds-model"
model_path = ls.options.pins.get_path(model_name)

# change tmp_dir to the desired directory
with tempfile.TemporaryDirectory() as tmp_dir:
    con = ls.connect()
    predict_xgb = con.register_xgb_model(model_name, model_path)
    cache = ParquetCacheStorage(source=con, path=tmp_dir)
    pg = ls.postgres.connect_examples()
    t = pg.table("diamonds").cache(storage=cache)

    output = t.mutate(prediction=predict_xgb.on_expr).pipe(ls.execute)
    print(output)
