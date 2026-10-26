import pyarrow as pa
import pytest
from pyiceberg.types import StringType

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import SourceCache
from xorq.catalog.drift import iter_leaf_reports
from xorq.catalog.enums import Verdict
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.refresh import op_key, refresh_build, refresh_schemas
from xorq.common.utils.graph_utils import replace_sources
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


def pinned_and_latest(con, table_name, df):
    con.create_table(table_name, df, overwrite=True)
    first_id = current_snapshot_id(con, table_name)
    con.insert(table_name, df)
    return {
        "first": con.table(table_name, snapshot_id=first_id),
        "latest": con.table(table_name),
    }


@pytest.mark.parametrize("which", ["first", "latest"])
def test_refresh_matches_loaded_iceberg_table(iceberg_con, quotes_df, tmp_path, which):
    expr = pinned_and_latest(iceberg_con, f"snapshot_refresh_{which}", quotes_df)[which]
    loaded = load_expr(build_expr(expr, builds_dir=tmp_path))
    (leaf,) = loaded.op().find(ops.DatabaseTable)

    refreshed = refresh_schemas(loaded, {op_key(leaf): leaf.schema})

    n = len(quotes_df)
    assert len(refreshed.execute()) == {"first": n, "latest": 2 * n}[which]


def test_source_cache_keeps_pinned_and_latest_apart(iceberg_con, quotes_df):
    exprs = pinned_and_latest(iceberg_con, "snapshot_source_cache", quotes_df)
    cache = SourceCache.from_kwargs(source=iceberg_con)

    actual = {key: len(expr.cache(cache).execute()) for key, expr in exprs.items()}

    n = len(quotes_df)
    assert actual == {"first": n, "latest": 2 * n}


def pinned_before_a_new_column(con, table_name, df):
    con.create_table(table_name, df, overwrite=True)
    first_id = current_snapshot_id(con, table_name)
    ice_table = con.catalog.load_table(f"{con.namespace}.{table_name}")
    with ice_table.update_schema() as update:
        update.add_column("note", StringType())
    con.insert(table_name, df.assign(note="added"))
    return con.table(table_name, snapshot_id=first_id)


def test_pinned_table_carries_its_snapshot_schema(iceberg_con, quotes_df):
    pinned = pinned_before_a_new_column(iceberg_con, "snapshot_schema", quotes_df)

    actual = pinned.execute()

    assert tuple(pinned.schema()) == tuple(quotes_df.columns)
    assert tuple(actual.columns) == tuple(quotes_df.columns)
    assert len(actual) == len(quotes_df)


def test_pinned_table_does_not_drift_on_a_new_column(iceberg_con, quotes_df, tmp_path):
    pinned = pinned_before_a_new_column(iceberg_con, "snapshot_drift", quotes_df)
    record = BuildRecord.from_build_dir(build_expr(pinned, builds_dir=tmp_path))

    (report,) = iter_leaf_reports(record)

    assert report.verdict == Verdict.EQUAL


def test_refresh_keeps_a_pinned_read_apart_from_the_current_one(
    iceberg_con, quotes_df, tmp_path
):
    table_name = "snapshot_refresh_both"
    exprs = pinned_and_latest(iceberg_con, table_name, quotes_df)
    expr = exprs["first"].select("symbol").union(exprs["latest"].select("symbol"))
    build_path = build_expr(expr, builds_dir=tmp_path)
    ice_table = iceberg_con.catalog.load_table(f"{iceberg_con.namespace}.{table_name}")
    with ice_table.update_schema() as update:
        update.add_column("note", StringType())

    refreshed = refresh_build(build_path)

    actual = {
        node.snapshot_id: tuple(node.schema)
        for node in refreshed.op().find(ops.DatabaseTable)
    }
    assert actual == {
        exprs["first"].op().snapshot_id: tuple(quotes_df.columns),
        None: (*quotes_df.columns, "note"),
    }


def test_replace_sources_transfers_a_loaded_iceberg_table(
    iceberg_con, quotes_df, tmp_path
):
    exprs = pinned_and_latest(iceberg_con, "snapshot_replace", quotes_df)
    loaded = load_expr(build_expr(exprs["latest"], builds_dir=tmp_path))
    (source,) = loaded.ls.backends
    con = xo.connect()

    moved = replace_sources({id(source): con}, loaded, transfer_tables=True)

    assert len(moved.execute()) == 2 * len(quotes_df)


def test_replace_sources_refuses_a_pinned_iceberg_table(iceberg_con, quotes_df):
    exprs = pinned_and_latest(iceberg_con, "snapshot_replace_pinned", quotes_df)
    con = xo.connect()

    with pytest.raises(ValueError, match="snapshot"):
        replace_sources({id(iceberg_con): con}, exprs["first"], transfer_tables=True)
