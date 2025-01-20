import letsql as ls
from letsql.expr.relations import into_backend

pg = ls.postgres.connect_env()
db = ls.duckdb.connect()


batting = pg.table("batting")
awards_players = db.read_parquet(
    ls.config.options.pins.get_path("awards_players"),
    table_name="awards_players",
)
left = batting.filter(batting.yearID == 2015)
right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
expr = left.join(into_backend(right, pg), ["playerID"], how="semi")[["yearID", "stint"]]


result = ls.execute(expr)
print(result)
