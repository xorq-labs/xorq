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
