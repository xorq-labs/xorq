from pathlib import Path

import letsql as ls


pg = ls.postgres.connect_examples()
con = ls.connect()  # empty connection
storage = ls.common.caching.ParquetCacheStorage(
    source=con,
    path=Path.cwd(),
)


alltypes = con.register(
    pg.table("functional_alltypes"), table_name="pg_functional_alltypes"
)
cached = (
    alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col).cache(
        storage=storage
    )  # cache expression (this creates a local table)
)
expr = cached.filter(
    [
        cached.float_col > 0,
        cached.smallint_col > 4,
        cached.int_col < cached.float_col * 2,
    ]
)
path = storage.get_loc(cached.ls.get_key())


print(f"{path} exists?: {path.exists()}")
result = cached.execute()  # the filter is executed on the local table
print(f"{path} exists?: {path.exists()}")
print(result)
