import pyarrow as pa

import xorq as xo
from xorq.backends.pyiceberg.tests.conftest import QUOTES_TABLE_NAME
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


def test_list_snapshots(iceberg_con):
    actual = iceberg_con.list_snapshots()
    assert isinstance(actual, dict)


def test_filter_by_snapshot_simple(iceberg_con):
    lookup = iceberg_con.list_snapshots()
    first, *_ = lookup[QUOTES_TABLE_NAME]

    table = iceberg_con.table(QUOTES_TABLE_NAME, snapshot_id=first)
    assert not table.execute().empty


def test_filter_by_snapshot(iceberg_con, quotes_df):
    table_name = "double_quotes"
    iceberg_con.create_table(table_name, quotes_df)
    quotes_double_bid = quotes_df.assign(bid=quotes_df.bid * 2)
    iceberg_con.insert(table_name, quotes_double_bid)
    lookup = iceberg_con.list_snapshots()
    first_id, second_id = lookup[table_name]

    first = iceberg_con.table(table_name, snapshot_id=first_id).execute()
    actual = iceberg_con.table(table_name, snapshot_id=second_id).execute()

    assert not first["bid"].equals(actual["bid"])
