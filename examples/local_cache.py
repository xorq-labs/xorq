from pathlib import Path

import ibis
import letsql as ls

pg = ibis.postgres.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    database="ibis_testing",
)

con = ls.connect()  # empty connection

alltypes = con.register(
    pg.table("functional_alltypes"), table_name="pg_functional_alltypes"
)

expr = alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)

storage = ls.common.caching.ParquetCacheStorage(
    source=con,
    path=Path.cwd(),
)

cached = expr.cache(storage=storage)  # cache expression (this creates a local table)

cached = cached.filter(
    [
        cached.float_col > 0,
        cached.smallint_col > 4,
        cached.int_col < cached.float_col * 2,
    ]
)


result = cached.execute()  # the filter is executed on the local table

print(result)
