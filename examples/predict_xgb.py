import tempfile

import xorq as xo
from xorq.common.caching import ParquetCacheStorage


model_name = "diamonds-model"
model_path = xo.options.pins.get_path(model_name)

# change tmp_dir to the desired directory
with tempfile.TemporaryDirectory() as tmp_dir:
    con = xo.connect()
    predict_xgb = con.register_xgb_model(model_name, model_path)
    cache = ParquetCacheStorage(source=con, path=tmp_dir)
    pg = xo.postgres.connect_examples()
    t = pg.table("diamonds").cache(storage=cache)

    output = t.mutate(prediction=predict_xgb.on_expr).pipe(xo.execute)
    print(output)
