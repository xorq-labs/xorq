import ibis

import letsql as ls

con = ls.connect()  # empty connection

con.add_connection(
    ibis.postgres.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="ibis_testing",
    )
)  # add Postgres connection

batting = con.table("batting")
awards_players = con.table("awards_players")

left = batting[batting.yearID == 2015]
right_df = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID").execute()

right = con.register(right_df, table_name="right")

expr = left.join(right, ["playerID"], how="semi")

result = expr.execute()
print(result)
