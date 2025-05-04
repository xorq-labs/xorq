import pyarrow as pa

import xorq as xo


if __name__ == "__main__":
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
    result = con.tables["inhabitants"].head().execute()
