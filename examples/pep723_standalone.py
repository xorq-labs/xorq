# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0"]
# ///
"""Standalone PEP 723 script — no pyproject.toml required.

Demonstrates building an xorq expression from a single .py file whose
dependencies are declared inline via PEP 723 metadata.  Build with:

    xorq uv build examples/pep723_standalone.py --pep723

The packager reads the ``# /// script`` block above, synthesises an
ephemeral project, resolves dependencies with ``uv lock``, and produces
a wheel — no surrounding project structure needed.
"""

import pandas as pd

import xorq.api as xo


con = xo.connect()
t = con.create_table(
    "iris",
    pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.4, 5.8, 5.0],
            "sepal_width": [3.5, 3.0, 3.2, 3.2, 2.7, 3.4],
            "species": [
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
            ],
        }
    ),
)

expr = t.filter([xo._.sepal_length > 5]).group_by("species").agg(xo._.sepal_width.sum())


if __name__ == "__pytest_main__":
    res = expr.execute()
    print(res)
    pytest_examples_passed = True
