"""Demonstrates ParquetCache for caching window function results computed in postgres.

Traditional approach: You would write SQL with window functions, execute it against
postgres, manually save the results to a file, and rebuild that cache whenever the
query changes. Tracking which cached file corresponds to which version of the query
is entirely your responsibility.

With xorq: Window functions are composed as Ibis expressions, and .cache() derives
the cache key from the expression itself. When the expression changes, stale results
are automatically invalidated, so you never serve outdated data.
"""
import xorq.api as xo
from xorq.api import _
from xorq.caching import ParquetCache


pg = xo.postgres.connect_examples()
con = xo.connect()
cache = ParquetCache.from_kwargs(source=con)


expr = (
    pg.table("batting")
    .mutate(row_number=xo.row_number().over(group_by=[_.playerID], order_by=[_.yearID]))
    .filter(_.row_number == 1)
    .cache(cache=cache)
)


if __name__ in ("__pytest_main__", "__main__"):
    print(f"{expr.ls.get_key()} exists?: {expr.ls.exists()}")
    res = xo.execute(expr)
    print(res)
    print(f"{expr.ls.get_key()} exists?: {expr.ls.exists()}")
    pytest_examples_passed = True
