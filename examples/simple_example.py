import pathlib

import letsql as ls


con = ls.connect()

iris_data_path = pathlib.Path(__file__).absolute().parent / "data" / "iris.csv"
iris = con.read_csv(iris_data_path, "iris")

res = (
    iris.filter([iris.sepal_length > 5])
    .group_by("species")
    .agg(iris.sepal_width.sum())
    .execute()
)

print(res)
