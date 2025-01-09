import letsql as ls
from letsql.common.caching import SourceStorage
from letsql.expr.relations import into_backend

con = ls.connect()
pg = ls.postgres.connect_env()

t = into_backend(pg.table("batting"), con, "ls_batting")

expr = (
    t.join(t, "playerID")
    .limit(15)
    .select(player_id="playerID", year_id="yearID_right")
    .cache(SourceStorage(source=con))
)

print(ls.execute(expr))
print(con.list_tables())
