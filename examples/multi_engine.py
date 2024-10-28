import letsql as ls


pg = ls.postgres.connect_examples()
db = ls.duckdb.connect()


batting = pg.table("batting")
awards_players = db.register(
    ls.config.options.pins.get_path("awards_players"),
    table_name="awards_players",
)
left = batting[batting.yearID == 2015]
right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")
expr = left.join(right, ["playerID"], how="semi")[["yearID", "stint"]]


result = ls.execute(expr)
print(result)
