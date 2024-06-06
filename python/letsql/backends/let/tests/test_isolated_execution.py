import ibis

import letsql as ls
from letsql.common.caching import ParquetCacheStorage


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


def test_multi_engine_cache(pg, tmp_path):
    con = ls.connect()
    db_con = ibis.duckdb.connect()

    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID > 2014].pipe(
        con.register, f"pg-{table_name}"
    )
    db_t = db_con.register(pg.table(table_name).to_pyarrow(), f"{table_name}")[
        lambda t: t.stint == 1
    ].pipe(con.register, f"db-{table_name}")

    expr = pg_t.join(
        db_t,
        db_t.columns,
    ).cache(storage=ParquetCacheStorage(tmp_path, con))
    expr.execute()
