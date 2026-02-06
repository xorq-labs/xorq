"""Transfers data between backends (Postgres to DuckDB) with joins and caching.

Traditional approach: You would extract data from Postgres into a pandas DataFrame, then
load it into DuckDB manually using its own API. Joins would happen in application code or
require duplicating data across databases, and caching intermediate results means writing
custom serialization logic.

With xorq: .into_backend() seamlessly moves data between engines in a single method call.
Joins across backends work automatically, and .cache() with SourceCache persists intermediate
results so repeated runs skip redundant computation.
"""
import xorq.api as xo
from xorq.caching import SourceCache


con = xo.connect()
pg = xo.postgres.connect_examples()


t = pg.table("batting").filter(xo._.yearID == 2015).into_backend(con, "ls_batting")
expr = (
    t.join(t, "playerID")
    .limit(15)
    .select(player_id="playerID", year_id="yearID_right")
    .cache(SourceCache.from_kwargs(source=con))
)


if __name__ in ("__pytest_main__", "__main__"):
    print(expr)
    print(expr.execute())
    print(con.list_tables())
    pytest_examples_passed = True
