import xorq as xo
from xorq.common.utils.defer_utils import deferred_read_parquet


awards_players_url = "https://storage.googleapis.com/letsql-pins/awards_players/20240711T171119Z-886c4/awards_players.parquet"
batting_url = "https://storage.googleapis.com/letsql-pins/batting/20240711T171118Z-431ef/batting.parquet"


con = xo.connect()
db = xo.duckdb.connect()


batting = deferred_read_parquet(
    path=batting_url,
    con=con,
    table_name="batting",
)
awards_players = deferred_read_parquet(
    path=awards_players_url,
    con=db,
    table_name="awards_players",
)


left = batting.filter(batting.yearID == 2015)
right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
expr = left.join(
    right.into_backend(con, "awards_players-filtered"),
    ["playerID"],
    how="semi",
)[["playerID", "yearID", "stint", "teamID", "lgID"]]
