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
