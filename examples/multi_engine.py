"""Joins data across multiple backends (Postgres and DuckDB) in a single expression.

Traditional approach: You would query each database separately, extract results into pandas
DataFrames, then merge them in Python. This loses query pushdown, requires manual data
movement, and makes it hard to optimize execution.

With xorq: Cross-backend joins are transparent. You write a single join expression and xorq
handles moving data between engines via into_backend, picking the optimal execution engine.
The query reads naturally even though the underlying data lives in different databases.
"""
import xorq.api as xo
from libs.postgres_helpers import connect_postgres
from xorq.expr.relations import into_backend


pg = connect_postgres()
db = xo.duckdb.connect()

batting = pg.table("batting")

awards_players = xo.examples.awards_players.fetch(backend=db)
left = batting.filter(batting.yearID == 2015)
right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
expr = left.join(into_backend(right, pg), ["playerID"], how="semi")[["yearID", "stint"]]


if __name__ in ("__pytest_main__", "__main__"):
    result = xo.execute(expr)
    print(result)
    pytest_examples_passed = True
