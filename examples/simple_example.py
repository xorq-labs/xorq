"""Loads the iris dataset, filters by sepal length, groups by species, and aggregates sepal width.

Traditional approach: You would use pandas to read a CSV with read_csv, filter rows
with boolean indexing, call groupby on the species column, and aggregate with .sum().
This locks you into pandas and requires eager execution of every step.

With xorq: The same Ibis expressions work across DuckDB, Postgres, and other backends
without changing your code. Execution is deferred until you call .execute(), and
results can be cached automatically to avoid redundant computation.
"""

import xorq.api as xo


expr = (
    xo.examples.iris.fetch(backend=xo.connect())
    .filter([xo._.sepal_length > 5])
    .group_by("species")
    .agg(xo._.sepal_width.sum())
)


if __name__ in ("__pytest_main__", "__main__"):
    res = expr.execute()
    print(res)
    pytest_examples_passed = True
