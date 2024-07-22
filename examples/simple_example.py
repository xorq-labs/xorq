import letsql as ls


con = ls.connect()
iris = con.read_csv(ls.config.options.pins.get_path("iris"), "iris")
res = (
    iris.filter([iris.sepal_length > 5])
    .group_by("species")
    .agg(iris.sepal_width.sum())
    .execute()
)

print(res)
