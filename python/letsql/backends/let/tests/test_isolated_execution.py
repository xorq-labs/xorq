import ibis

import letsql as ls


def test_multiple_pipes(pg):
    """This test address the issue reported on bug #69
    link: https://github.com/letsql/letsql/issues/69

    NOTE
    The previous tests didn't catch it because the con object registered the table batting.
    """

    con = ls.connect()
    duckdb_con = ibis.duckdb.connect()
    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015].pipe(
        con.register, f"pg-{table_name}"
    )
    db_t = duckdb_con.register(pg_t.to_pyarrow(), f"{table_name}")[
        lambda t: t.yearID == 2014
    ].pipe(con.register, f"db-{table_name}")

    expr = pg_t.join(
        db_t,
        "playerID",
    )

    expr.execute()
