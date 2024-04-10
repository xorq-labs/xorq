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

alltypes = con.table("functional_alltypes")

expr = alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
cached = expr.cache()  # cache expression (this creates a local table)

cached = cached.filter(
    [
        cached.float_col > 0,
        cached.smallint_col > 4,
        cached.int_col < cached.float_col * 2,
    ]
)


result = cached.execute()  # the filter is executed on the local table

print(result)
