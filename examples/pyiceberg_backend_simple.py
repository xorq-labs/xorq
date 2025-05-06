import pyarrow as pa

import xorq as xo


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
