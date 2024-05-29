import pathlib

import ibis

import letsql as ls

con = ls.connect()  # empty connection

pg = ibis.postgres.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    database="ibis_testing",
)

root = pathlib.Path(__file__).absolute().parents[1]
path = root / "ci" / "ibis-testing-data" / "parquet" / "awards_players.parquet"
batting = con.register(pg.table("batting"), table_name="batting")

awards_players = con.register(path, table_name="awards_players")

left = batting[batting.yearID == 2015]
right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID").execute()

expr = left.join(right, ["playerID"], how="semi")

result = expr.execute()

print(result)
