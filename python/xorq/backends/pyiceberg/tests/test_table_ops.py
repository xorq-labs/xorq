import pyarrow as pa
import pytest

from xorq import Schema
from xorq.backends.pyiceberg.tests.conftest import QUOTES_TABLE_NAME


def test_create_table_from_expr(iceberg_con, quotes_table):
    t = quotes_table.select("timestamp", "bid", "ask").limit(10).as_table()

    actual_t = iceberg_con.create_table("selected_quotes", t)

    assert not actual_t.execute().empty


def identity(val):
    return val


def execute(val):
    val = val.execute()
    val["timestamp"] = val["timestamp"].astype("datetime64[us]")
    return val


@pytest.mark.parametrize(
    "fun",
    [
        pytest.param(execute, id="pandas"),
        pytest.param(identity, id="xorq"),
    ],
)
def test_append(iceberg_con, trades_df, fun):
    table_name = "trades"

    t = iceberg_con.create_table(table_name, trades_df, overwrite=True)
    assert table_name in iceberg_con.list_tables()
    expected = t.execute()

    iceberg_con.insert(table_name, fun(t))
    actual = t.execute()
    assert len(actual) == len(expected) * 2


def test_create_empty_raises(iceberg_con):
    with pytest.raises(NotImplementedError):
        schema = Schema.from_pyarrow(
            pa.schema(
                [
                    pa.field("question", pa.string()),
                    pa.field("product_description", pa.string()),
                    pa.field("image_url", pa.string()),
                    pa.field("label", pa.uint8(), nullable=True),
                ]
            )
        )
        iceberg_con.create_table("empty", schema=schema)


def test_read_record_batches(iceberg_con, quotes_table):
    table_name = "quotes_records"
    t = iceberg_con.read_record_batches(
        quotes_table.to_pyarrow_batches(), table_name=table_name
    )
    assert table_name in iceberg_con.list_tables()
    assert not t.execute().empty


def test_get_schema(iceberg_con):
    iceberg_con.get_schema(QUOTES_TABLE_NAME)
