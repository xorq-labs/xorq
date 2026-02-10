"""Loads the iris dataset and caches filtered results to Parquet using ParquetCache.

Traditional approach: You would use pd.read_csv to load the data, apply filters in pandas,
then manually cache results by writing to a Parquet file. Checking whether the cache is
still valid requires hand-rolled file-existence checks and invalidation logic.

With xorq: .cache() with ParquetCache adds input-addressed caching in one line. The cache
key is derived from the expression itself, so changes to filters automatically invalidate
stale results without any manual bookkeeping.
"""
from pathlib import Path

import xorq.api as xo
from xorq.caching import ParquetCache


t = xo.examples.iris.fetch()
con = t.op().source
cache = ParquetCache.from_kwargs(source=con, relative_path=Path("./parquet-cache"))

expr = t.filter([t.species == "Setosa"]).cache(cache=cache)


if __name__ == "__pytest_main__":
    (op,) = expr.ls.cached_nodes
    path = cache.storage.get_path(op.to_expr().ls.get_key())
    print(f"{path} exists?: {path.exists()}")
    result = xo.execute(expr)
    print(f"{path} exists?: {path.exists()}")
    pytest_examples_passed = True
