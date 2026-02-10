"""Creates and queries PyIceberg tables using xorq's Iceberg backend integration.

Traditional approach: You would manually set up an Iceberg catalog, define table schemas
through the Iceberg API, and write data using Iceberg's own write path. Querying requires
separate reader setup, and mixing Iceberg tables with other data sources means glue code
between incompatible APIs.

With xorq: PyIceberg is integrated as a first-class backend, so you can use familiar Ibis
expressions to create and query Iceberg tables. The same expression syntax works across
Iceberg and other backends, making it easy to incorporate Iceberg into multi-engine workflows.
"""
import pyarrow as pa

import xorq.api as xo


con = xo.pyiceberg.connect()
arrow_schema = pa.schema(
    [
        pa.field("city", pa.string(), nullable=False),
        pa.field("inhabitants", pa.int32(), nullable=False),
    ]
)
df = pa.Table.from_pylist(
    [
        {"city": "Drachten", "inhabitants": 45505},
        {"city": "Berlin", "inhabitants": 3432000},
        {"city": "Paris", "inhabitants": 2103000},
    ],
    schema=arrow_schema,
)
inhabitants = con.create_table("inhabitants", df, overwrite=True)
expr = con.tables["inhabitants"].head()

if __name__ == "__pytest_main__":
    res = expr.execute()
    print(res)
    pytest_examples_passed = True
