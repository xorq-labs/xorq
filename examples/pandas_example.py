"""Registers a pandas DataFrame as a table in xorq and queries it with Ibis expressions.

Traditional approach: You would perform all transformations directly on the pandas
DataFrame using methods like .head(), .query(), or bracket indexing. This works fine
for small data but offers no path to switching engines or composing with other backends.

With xorq: Wrapping a pandas DataFrame in xorq gives you a SQL-like Ibis API on top
of it, so you can compose expressions, switch to DuckDB or Postgres without rewriting
logic, and benefit from deferred execution and caching.
"""

import pandas as pd

import xorq.api as xo


con = xo.connect()

df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
t = con.create_table("frame", df)
expr = t.head(3)


if __name__ == "__pytest_main__":
    res = expr.execute()
    print(res)
    pytest_examples_passed = True
