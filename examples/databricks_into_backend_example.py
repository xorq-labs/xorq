"""Move data between DuckDB and Databricks using into_backend.

Demonstrates pushing a local DuckDB table into Databricks, transforming
remotely, and pulling results back into DuckDB — all with a single
expression chain.

Install:
  uv pip install 'xorq[databricks]'
  curl -LsSf https://dbc.columnar.tech/install.sh | sh
  dbc install databricks

Requires environment variables:
  DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
"""

import xorq.api as xo


# Local DuckDB with a toy dataset
ddb = xo.duckdb.connect()
t = ddb.create_table(
    "players",
    {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [95, 82, 78, 91, 88],
        "team": ["A", "B", "A", "B", "A"],
    },
)

# Connect to Databricks
dbx = xo.databricks.connect()

# Push to Databricks, filter and mutate remotely, pull back to DuckDB
expr = (
    t.into_backend(dbx)
    .filter(xo._.score > 80)
    .mutate(grade=(xo._.score >= 90).ifelse("top", "mid"))
    .select("name", "score", "team", "grade")
    .into_backend(ddb)
)


if __name__ == "__pytest_main__":
    result = expr.execute()
    print("Players with score > 80 (round-tripped through Databricks):")
    print(result)
    pytest_examples_passed = True
