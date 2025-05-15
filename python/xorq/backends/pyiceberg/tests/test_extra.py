import pyarrow as pa

import xorq as xo
from xorq.caching import SourceStorage
from xorq.tests.util import assert_frame_equal


def test_into_backend(iceberg_con, trades_df):
    con = xo.connect()
    t = con.create_table("trades", trades_df)
    expr = t.select(t.symbol, t.price, t.volume).into_backend(
        iceberg_con, "xorq_trades"
    )

    expected = trades_df[["symbol", "price", "volume"]]

    actual = expr.execute()

    assert_frame_equal(actual, expected)


def test_out_into_backend(quotes_table, quotes_df):
    con = xo.connect()
    expr = quotes_table.select("symbol", "bid").into_backend(con, name="iceberg_quotes")
    expected = quotes_df[["symbol", "bid"]]
    actual = expr.execute()

    assert_frame_equal(actual, expected)


def test_caching(iceberg_con, quotes_table):
    storage = SourceStorage(source=iceberg_con)
    uncached_expr = quotes_table.select("symbol", "bid").filter(xo._.symbol == "GOOGL")
    expr = uncached_expr.cache(storage)

    assert not storage.exists(uncached_expr)
    assert not expr.execute().empty
    assert storage.exists(uncached_expr)


def test_upsert(iceberg_con, quotes_table):
    arrow_schema = pa.schema(
        [
            pa.field("city", pa.string(), nullable=False),
            pa.field("inhabitants", pa.int32(), nullable=False),
        ]
    )

    df = pa.Table.from_pylist(
        [
            {"city": "Amsterdam", "inhabitants": 921402},
            {"city": "San Francisco", "inhabitants": 808988},
            {"city": "Drachten", "inhabitants": 45019},
            {"city": "Paris", "inhabitants": 2103000},
        ],
        schema=arrow_schema,
    )

    table = iceberg_con.create_table("cities", df)
    before = table.execute()

    df = pa.Table.from_pylist(
        [
            {"city": "Drachten", "inhabitants": 45505},
            {"city": "Berlin", "inhabitants": 3432000},
            {"city": "Paris", "inhabitants": 2103000},
        ],
        schema=arrow_schema,
    )

    table = iceberg_con.upsert("cities", df, join_cols=["city"])
    after = table.execute()

    assert not before.equals(after)
    actual = after.set_index("city").to_dict()["inhabitants"]

    assert "Berlin" in actual
    assert actual["Drachten"] == 45505
