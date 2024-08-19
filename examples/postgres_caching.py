import letsql as ls
from letsql import _
from letsql.common.caching import ParquetCacheStorage

pg = ls.postgres.connect_examples()

for table_name in pg.list_tables():
    if table_name.startswith(ls.config.options.cache.key_prefix):
        pg.drop_table(table_name)

cache = ParquetCacheStorage(source=pg)

t = (
    pg.table("batting")
    .mutate(row_number=ls.row_number().over(group_by=[_.playerID], order_by=[_.yearID]))
    .filter(_.row_number == 1)
    .cache(storage=cache)
)
print(f"{t.ls.get_key()} exists?: {t.ls.exists()}")
res = t.execute()
print(res)
print(f"{t.ls.get_key()} exists?: {t.ls.exists()}")
