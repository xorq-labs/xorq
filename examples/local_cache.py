from pathlib import Path

import xorq.api as xo
from xorq.caching import ParquetCache


pg = xo.postgres.connect_examples()
con = xo.connect()
storage = ParquetCache.from_kwargs(
    source=con,
    relative_path=Path("./parquet-cache"),
)


cached = (
    pg.table("functional_alltypes")
    .into_backend(con)
    .select(xo._.smallint_col, xo._.int_col, xo._.float_col)
    .cache(storage=storage)
)
expr = cached.filter(
    [
        xo._.float_col > 0,
        xo._.smallint_col > 4,
        xo._.int_col < cached.float_col * 2,
    ]
)


if __name__ == "__pytest_main__":
    path = storage.get_loc(cached.ls.get_key())
    print(f"{path} exists?: {path.exists()}")
    result = xo.execute(expr)
    print(f"{path} exists?: {path.exists()}")
    print(result)
    pytest_examples_passed = True
