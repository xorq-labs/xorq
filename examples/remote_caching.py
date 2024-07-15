import pathlib

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

root = pathlib.Path(__file__).absolute().parents[1]
path = root / "ci" / "ibis-testing-data" / "parquet" / "batting.parquet"
right = ddb.read_parquet(path, table_name="batting")[lambda t: t.yearID == 2014].pipe(
    con.register, table_name="ddb-batting"
)

left = pg.table("batting")[lambda t: t.yearID == 2015].pipe(
    con.register, table_name="pg-batting"
)

expr = left.join(
    right,
    "playerID",
).cache(SourceStorage(pg))

res = expr.execute()
print(res)
