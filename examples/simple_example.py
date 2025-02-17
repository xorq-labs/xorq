import letsql as ls


con = ls.connect()
iris = ls.examples.iris.fetch(backend=con, table_name="iris")
res = (
    iris.filter([iris.sepal_length > 5])
    .group_by("species")
    .agg(iris.sepal_width.sum())
    .execute()
)

print(res)
