import xorq.api as xo
from xorq.caching import SourceCache


con = xo.connect()
pg = xo.postgres.connect_env()


t = pg.table("batting").filter(xo._.yearID == 2015).into_backend(con, "ls_batting")
expr = (
    t.join(t, "playerID")
    .limit(15)
    .select(player_id="playerID", year_id="yearID_right")
    .cache(SourceCache.from_kwargs(source=con))
)


if __name__ == "__pytest_main__":
    print(expr)
    print(expr.execute())
    print(con.list_tables())
    pytest_examples_passed = True
