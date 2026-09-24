"""Demonstrates Google Cloud Storage caching using GCSCache.

Traditional approach: You would query your data, manually upload the results as
parquet files to GCS using the google-cloud-storage library, and check blob
existence to determine cache hits. Cache key management and invalidation are
entirely your responsibility.

With xorq: GCSCache plugs into the same .cache() API as any other cache backend.
Swap ParquetCache for GCSCache and your results are cached in a GCS bucket with
the same input-addressed semantics -- no changes to your pipeline logic needed.
"""

import xorq.api as xo
from xorq.caching import GCSCache


bucket_name = "expr-cache"
con = xo.connect()
cache = GCSCache.from_kwargs(bucket_name=bucket_name, source=con)


expr = xo.deferred_read_csv(
    path=xo.options.pins.get_path("bank-marketing"),
    con=con,
).cache(cache=cache)


if __name__ == "__pytest_main__":
    # The cache key is input-addressed, so every run of this fixed expression
    # targets the same object. A run that dies between the write and the drop
    # below strands exactly that object and would leave the assertion that
    # follows failing for every later run, so clear it first.
    if expr.ls.cache_exists():
        cache.drop(expr)
    assert not expr.ls.cache_exists()
    df = expr.execute()
    assert expr.ls.cache_exists()
    key = cache.calc_key(expr)
    listing = cache.storage.fs.ls(cache.storage.get_path(key), detail=True)
    print(listing)
    cache.drop(expr)
    assert not expr.ls.cache_exists()
    pytest_examples_passed = True
