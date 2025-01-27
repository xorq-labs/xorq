from pathlib import Path

import letsql as ls
from letsql.common.caching import ParquetCacheStorage


t = ls.examples.penguins.fetch()

con = t.op().source
storage = ParquetCacheStorage(source=con, path=Path.cwd())

cached = t.filter([t.species == "Adelie"]).cache(storage=storage)
(op,) = cached.ls.cached_nodes
path = storage.get_loc(op.to_expr().ls.get_key())
print(f"{path} exists?: {path.exists()}")
result = ls.execute(cached)
print(f"{path} exists?: {path.exists()}")
