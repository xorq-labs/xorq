import pyarrow as pa
import pytest
from pyiceberg.catalog import load_catalog

import xorq.api as xo
import xorq.backends.pyiceberg as pyiceberg_backend
from xorq.backends.pyiceberg.tests.conftest import QUOTES_TABLE_NAME
from xorq.caching import ParquetCache
from xorq.vendor.ibis import Schema


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


@pytest.mark.parametrize(
    "fun",
    [
        pytest.param(execute, id="pandas"),
        pytest.param(identity, id="xorq"),
    ],
)
def test_insert_overwrite(iceberg_con, trades_df, fun):
    table_name = "trades_overwrite"

    t = iceberg_con.create_table(table_name, trades_df, overwrite=True)
    assert table_name in iceberg_con.list_tables()
    expected = t.execute()

    # overwrite with a differently-shaped, differently-valued dataset so the
    # test fails on a silent no-op (data unchanged) as well as on a double-append
    new_df = trades_df.head(5).assign(price=trades_df["price"].head(5) + 1000)
    src_name = "trades_overwrite_src"
    new_t = iceberg_con.create_table(src_name, new_df, overwrite=True)

    iceberg_con.insert(table_name, fun(new_t), mode="overwrite")
    actual = t.execute().sort_values("timestamp").reset_index(drop=True)

    assert len(actual) == len(new_df)
    assert len(actual) != len(expected)
    # every price reflects the +1000 offset → data was actually replaced
    assert (actual["price"] > 1000).all()


def test_insert_overwrite_atomic_on_schema_mismatch(iceberg_con, trades_df):
    # a failed overwrite (schema mismatch) must abort the whole delete+append
    # transaction, leaving the original data intact rather than truncated
    table_name = "trades_overwrite_atomic"

    t = iceberg_con.create_table(table_name, trades_df, overwrite=True)
    expected = t.execute()

    bad_df = trades_df.assign(extra=1)
    with pytest.raises(ValueError):
        iceberg_con.insert(table_name, bad_df, mode="overwrite")

    actual = t.execute()
    assert len(actual) == len(expected)


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


@pytest.fixture
def con_with_t_in_four_places(fresh_con):
    """``t`` in four places, with a distinct row count in each.

    1 row in the default namespace, 2 in namespace ``other``, 3 in catalog
    ``other``'s default namespace, and 4 in catalog ``other``'s namespace ``other``.
    """
    fresh_con.create_table("t", pa.table({"a": [1]}))
    fresh_con.catalog.create_namespace("other")
    fresh_con.create_table("t", pa.table({"a": [1, 2]}), database="other")
    other_catalog = load_catalog("other", **fresh_con.catalog_params)
    other_catalog.create_namespace(fresh_con.namespace)
    other_catalog.create_table(
        f"{fresh_con.namespace}.t", schema=pa.schema([("a", pa.int64())])
    ).append(pa.table({"a": [1, 2, 3]}))
    other_catalog.create_namespace("other")
    other_catalog.create_table("other.t", schema=pa.schema([("a", pa.int64())])).append(
        pa.table({"a": [1, 2, 3, 4]})
    )
    return fresh_con


# Backend.table names the iceberg catalog ``schema``
@pytest.mark.parametrize(
    "kwargs, rows",
    [
        pytest.param({}, 1, id="default"),
        pytest.param({"database": "other"}, 2, id="database"),
        pytest.param({"schema": "other"}, 3, id="catalog"),
        pytest.param({"schema": "other", "database": "other"}, 4, id="both"),
    ],
)
def test_table_reads_database_and_catalog(con_with_t_in_four_places, kwargs, rows):
    con = con_with_t_in_four_places

    assert len(con.table("t", **kwargs).execute()) == rows


def test_cache_key_tracks_table_database(con_with_t_in_four_places, tmp_path):
    con = con_with_t_in_four_places
    expr = con.table("t", database="other").cache(
        ParquetCache.from_kwargs(base_path=tmp_path)
    )
    before = len(expr.execute())
    con.insert("t", pa.table({"a": [3]}), database="other")

    assert (before, len(expr.execute())) == (2, 3)


def test_each_catalog_is_loaded_once(tmp_path, monkeypatch):
    # Records calls to the backend's load_catalog, so it tests how catalogs are
    # built rather than what a read returns. The number of open catalogs (and
    # their SQLAlchemy engines) is not observable through the backend, so this
    # is the only check that a lookup by name does not build a new one.
    loaded = []

    def recording_load_catalog(name, **properties):
        loaded.append(name)
        return load_catalog(name, **properties)

    monkeypatch.setattr(pyiceberg_backend, "load_catalog", recording_load_catalog)
    con = xo.pyiceberg.connect(warehouse_path=str(tmp_path / "warehouse"))
    con.create_table("t", pa.table({"a": [1]}))
    other_catalog = load_catalog("other", **con.catalog_params)
    other_catalog.create_namespace(con.namespace)
    other_catalog.create_table(
        f"{con.namespace}.t", schema=pa.schema([("a", pa.int64())])
    )
    for catalog in ("other", "other", "default"):
        con.table("t", schema=catalog).execute()

    assert loaded == ["default", "other"]
