import letsql as ls
from letsql.expr.relations import into_backend


pg = ls.postgres.connect_env()
db = ls.duckdb.connect()

batting = pg.table("batting")

awards_players = ls.examples.awards_players.fetch(backend=db)
left = batting.filter(batting.yearID == 2015)
right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
expr = left.join(into_backend(right, pg), ["playerID"], how="semi")[["yearID", "stint"]]


result = ls.execute(expr)
print(result)
