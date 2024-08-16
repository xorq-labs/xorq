import letsql as ls


con = ls.connect()  # empty connection
pg = ls.postgres.connect_examples()
db = ls.duckdb.connect()


batting = pg.table("batting").pipe(con.register, table_name="batting")
awards_players = db.register(
    ls.config.options.pins.get_path("awards_players"),
    table_name="awards_players",
).pipe(con.register, "db-awards_players")


left = batting[batting.yearID == 2015]
right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")
expr = left.join(right, ["playerID"], how="semi")
result = expr.execute()


print(result)
print(tuple(dt.args for dt in expr.ls.native_dts))
