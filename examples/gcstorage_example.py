import xorq as xo
from xorq.caching import GCStorage


bucket_name = "expr-cache"
con = xo.connect()
storage = GCStorage(bucket_name=bucket_name)


expr = xo.deferred_read_csv(
    path=xo.options.pins.get_path("bank-marketing"),
    con=con,
).cache(storage=storage)


if __name__ == "__pytest_main__":
    assert not expr.ls.exists()
    df = expr.execute()
    assert expr.ls.exists()
    listing = storage.cache.storage.fs.ls(storage.get_path(expr), detail=True)
    print(listing)
    storage.cache.drop(expr)
    assert not expr.ls.exists()
    pytest_examples_passed = True
