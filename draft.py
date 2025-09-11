import xorq.api as xo
from xorq.caching import ParquetStorage, SourceStorage


expr = (
    xo.read_parquet(
        "/home/daniel/PycharmProjects/xorq/ci/ibis-testing-data/parquet/astronauts.parquet",
        table_name="astronauts",
    )
    .select("id", "number", "name", "nationwide_number")
    .limit(100)
    .cache(SourceStorage(source=xo.duckdb.connect()))
    .distinct(on=("name",))
    .order_by("number")
    .cache(ParquetStorage(source=xo.connect()))
)
print(expr.execute())
