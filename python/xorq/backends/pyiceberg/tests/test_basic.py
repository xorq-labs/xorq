import pytest

from xorq.backends.pyiceberg.tests.conftest import QUOTES_TABLE_NAME
from xorq.tests.util import assert_frame_equal


def test_create_table(iceberg_con, quotes_df):
    # the iceberg_con fixture creates the table
    assert QUOTES_TABLE_NAME in iceberg_con.list_tables()

    t = iceberg_con.table(QUOTES_TABLE_NAME)
    assert_frame_equal(t.execute(), quotes_df, check_dtype=False)


def test_create_and_drop_table(iceberg_con, trades_df):
    table_name = "trades"

    iceberg_con.create_table(table_name, trades_df, overwrite=True)
    assert table_name in iceberg_con.list_tables()

    iceberg_con.drop_table(table_name)
    assert table_name not in iceberg_con.list_tables()


def test_select(quotes_table):
    expected = quotes_table.execute()["symbol"].to_frame()
    actual = quotes_table.select("symbol").execute()

    assert_frame_equal(actual, expected, check_dtype=False)


def test_limit(quotes_table):
    actual = quotes_table.limit(5).execute()
    assert len(actual) == 5


def test_filter(quotes_table):
    df = quotes_table.execute()
    expected = df[df["symbol"] == "GOOGL"]
    actual = quotes_table.filter(quotes_table.symbol == "GOOGL").execute()

    assert_frame_equal(actual, expected, check_dtype=False)


def test_full_selection(quotes_table):
    actual = (
        quotes_table.select("symbol", "bid")
        .filter(quotes_table.symbol == "GOOGL")
        .limit(5)
        .execute()
    )
    assert len(actual) == 5
    assert actual.columns.tolist() == ["symbol", "bid"]
    assert actual["symbol"].eq("GOOGL").sum() == 5


def test_raises_not_implemented_error(quotes_table):
    with pytest.raises(NotImplementedError):
        quotes_table.count().execute()
