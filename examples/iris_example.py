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
