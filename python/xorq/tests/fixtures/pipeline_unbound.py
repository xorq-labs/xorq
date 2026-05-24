import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet


data_url = "https://storage.googleapis.com/letsql-pins/penguins/20250703T145709Z-c3cde/penguins.parquet"

con = xo.duckdb.connect()

source = deferred_read_parquet(
    con=con,
    path=data_url,
    table_name="penguins",
).tag("source")

expr = source.filter(source.species == "Adelie").select(
    "species", "island", "bill_length_mm"
)
