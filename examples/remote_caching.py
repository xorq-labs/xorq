import letsql as ls
from letsql.common.caching import SourceStorage

con = ls.connect()
ddb = ls.duckdb.connect()
pg = ls.postgres.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    database="ibis_testing",
)

name = "batting"
path = ls.config.options.pins.get_path(name)

right = ddb.read_parquet(path, table_name=name)[lambda t: t.yearID == 2014].pipe(
    con.register, table_name=f"ddb-{name}"
)

left = pg.table(name)[lambda t: t.yearID == 2015].pipe(
    con.register, table_name=f"pg-{name}"
)

expr = left.join(
    right,
    "playerID",
).cache(SourceStorage(source=pg))

res = expr.execute()
print(res)
