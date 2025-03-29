import xorq as xo


con = xo.connect()
iris = xo.examples.iris.fetch(backend=con, table_name="iris")
res = (
    iris.filter([iris.sepal_length > 5])
    .group_by("species")
    .agg(iris.sepal_width.sum())
    .execute()
)

print(res)
pytest_examples_passed = True
