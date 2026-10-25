import pyarrow as pa

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import SourceCache
from xorq.ibis_yaml.compiler import build_expr, load_expr
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
    cache = SourceCache.from_kwargs(source=iceberg_con)
    uncached_expr = quotes_table.select("symbol", "bid").filter(xo._.symbol == "GOOGL")
    expr = uncached_expr.cache(cache)

    assert not cache.exists(uncached_expr)
    assert not expr.execute().empty
    assert cache.exists(uncached_expr)


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


def current_snapshot_id(con, table_name):
    ice_table = con.catalog.load_table(f"{con.namespace}.{table_name}")
    return ice_table.current_snapshot().snapshot_id


def test_filter_by_snapshot(iceberg_con, quotes_df):
    table_name = "snapshot_rows"
    iceberg_con.create_table(table_name, quotes_df, overwrite=True)
    first_id = current_snapshot_id(iceberg_con, table_name)
    iceberg_con.insert(table_name, quotes_df)
    second_id = current_snapshot_id(iceberg_con, table_name)
    iceberg_con.insert(table_name, quotes_df)
    exprs = {
        "latest": iceberg_con.table(table_name),
        "first": iceberg_con.table(table_name, snapshot_id=first_id),
        "second": iceberg_con.table(table_name, snapshot_id=second_id),
        "hand_built": ops.DatabaseTable(
            name=table_name,
            schema=iceberg_con.get_schema(table_name),
            source=iceberg_con,
        ).to_expr(),
    }

    actual = {key: len(expr.execute()) for key, expr in exprs.items()}

    n = len(quotes_df)
    assert actual == {"latest": 3 * n, "first": n, "second": 2 * n, "hand_built": 3 * n}


def test_snapshot_pin_survives_build_and_load(iceberg_con, quotes_df, tmp_path):
    table_name = "snapshot_build"
    iceberg_con.create_table(table_name, quotes_df, overwrite=True)
    first_id = current_snapshot_id(iceberg_con, table_name)
    iceberg_con.insert(table_name, quotes_df)
    paths = {
        "first": build_expr(
            iceberg_con.table(table_name, snapshot_id=first_id), builds_dir=tmp_path
        ),
        "latest": build_expr(iceberg_con.table(table_name), builds_dir=tmp_path),
    }

    actual = {key: len(load_expr(path).execute()) for key, path in paths.items()}

    n = len(quotes_df)
    assert actual == {"first": n, "latest": 2 * n}
    assert paths["first"].name != paths["latest"].name
