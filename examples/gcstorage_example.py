import xorq.api as xo
from xorq.caching import GCSCache


bucket_name = "expr-cache"
con = xo.connect()
cache = GCSCache.from_kwargs(bucket_name=bucket_name, source=con)


expr = xo.deferred_read_csv(
    path=xo.options.pins.get_path("bank-marketing"),
    con=con,
).cache(cache=cache)


if __name__ in ("__pytest_main__", "__main__"):
    assert not expr.ls.exists()
    df = expr.execute()
    assert expr.ls.exists()
    listing = cache.cache.cache.fs.ls(cache.get_path(expr), detail=True)
    print(listing)
    cache.cache.drop(expr)
    assert not expr.ls.exists()
    pytest_examples_passed = True
