import pandas as pd

import xorq as xo
from xorq.flight import FlightServer, FlightUrl


def get_expr(con0, con1):
    df_groups = {
        "a": pd.DataFrame(
            {"time": [1, 3, 5]},
        ),
        "b": pd.DataFrame(
            {"time": [2, 4, 6]},
        ),
        "c": pd.DataFrame(
            {"time": [2.5, 4.5, 6.5]},
        ),
    }

    for name, df in df_groups.items():
        con0.register(df, f"df-{name}")
    dct = {
        table: con0.table(table).into_backend(con1, f"remote-{table}")
        for table in reversed(list(con0.tables))
    }
    (t, other, *others) = tuple(dct.values())
    for other in others[:1]:
        t = t.asof_join(other, "time").drop("time_right")

    return t


ddb1 = xo.duckdb.connect()
ddb2 = xo.duckdb.connect()

ddb_expr = get_expr(ddb1, ddb2)

# with DuckDB is empty
assert ddb_expr.execute().empty

url0 = FlightUrl(port=5005)
url1 = FlightUrl(port=5438)
with FlightServer(
    flight_url=url0,
    verify_client=False,
    connection=xo.duckdb.connect,
) as main0:
    with FlightServer(
        flight_url=url1,
        verify_client=False,
        connection=xo.duckdb.connect,
    ) as main1:
        flight_expr = get_expr(main0.con, main1.con)

        # with FlightServer wrapper on DuckDB is not empty
        assert not flight_expr.execute().empty
