import letsql as ls
from letsql import _
from letsql.common.caching import SourceStorage


con = ls.connect()
ddb = ls.duckdb.connect()
pg = ls.postgres.connect_env()

name = "batting"

right = (
    ls.examples.get_table_from_name(name, backend=ddb)
    .filter(_.yearID == 2014)
    .pipe(con.register, table_name=f"ddb-{name}")
)

left = (
    pg.table(name).filter(_.yearID == 2015).pipe(con.register, table_name=f"pg-{name}")
)

expr = left.join(
    right,
    "playerID",
).cache(SourceStorage(source=pg))

res = ls.execute(expr)
print(res)
