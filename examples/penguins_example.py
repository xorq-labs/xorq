import letsql as ls
from letsql.common.caching import ParquetCacheStorage
from pathlib import Path

t = ls.examples.penguins.fetch()

con = t.op().source

t.filter([t.species == "Adelie"]).cache(
    storage=ParquetCacheStorage(source=con, path=Path.cwd())
).execute()
