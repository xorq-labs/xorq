"""Demonstrates remote caching by using DuckDB as a data source and postgres as the cache backend.

Traditional approach: You would query DuckDB locally, then manually write the
results into postgres tables so other services can access them. Keeping schemas
in sync between the two systems and deciding when to refresh the postgres copy
is tedious and error-prone.

With xorq: SourceCache with a postgres backend automatically materializes DuckDB
query results into postgres tables, keyed by expression content. Schema management
and cache invalidation are handled for you.
"""

import xorq.api as xo
from xorq.api import _
from xorq.caching import SourceCache


con = xo.connect()
ddb = xo.duckdb.connect()
pg = xo.postgres.connect_env()

name = "batting"

right = (
    xo.examples.get_table_from_name(name, backend=ddb)
    .filter(_.yearID == 2014)
    .into_backend(con)
)
left = pg.table(name).filter(_.yearID == 2015).into_backend(con)

expr = left.join(
    right,
    "playerID",
).cache(SourceCache.from_kwargs(source=pg))


if __name__ == "__pytest_main__":
    res = expr.execute()
    print(res)
    pytest_examples_passed = True
