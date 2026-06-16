import warnings

import pandas as pd
import pyarrow as pa
import pytest
from batchcorder import StreamCache

import xorq.api as xo
from xorq.backends.xorq_datafusion import _select_and_cast


def test_select_and_cast_missing_raises():
    batch = pa.record_batch({"a": [1, 2]})
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    with pytest.raises(ValueError, match="batch schema mismatch"):
        _select_and_cast(batch, schema)


def test_select_and_cast_extra_warns_and_drops():
    batch = pa.record_batch({"a": [1, 2], "b": [3, 4], "extra": [5, 6]})
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _select_and_cast(batch, schema)
    assert len(w) == 1
    assert "extra" in str(w[0].message)
    assert result.schema == schema
    assert result.num_rows == 2


def test_select_and_cast_exact_schema():
    batch = pa.record_batch({"a": [1, 2], "b": [3, 4]})
    schema = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    result = _select_and_cast(batch, schema)
    assert result.schema == schema
    assert result.num_rows == 2


def test_no_name_register():
    xo.connect().register(pd.DataFrame({"a": [1]}))


def test_register_memtable():
    con = xo.connect()
    # memtable: no backends, not unbound → execute() → pa.Table path (not IbisTableProvider)
    t = con.register(xo.memtable({"a": [1, 2, 3]}), table_name="from_memtable")
    assert "from_memtable" in con.list_tables()
    assert t.count().execute() == 3


def test_register_table_provider():
    con = xo.connect()
    con.create_table("src", pa.table({"a": [1, 2, 3]}))
    source = con.table("src")
    # explicit name lands in the catalog
    con.register_table_provider(source, table_name="via_provider")
    assert "via_provider" in con.list_tables()
    # None table_name generates a name; returned table reflects it
    t = con.register_table_provider(source)
    assert t.get_name() in con.list_tables()


def test_read_record_batches_type_mismatch() -> None:
    con = xo.connect()
    utf8_schema = pa.schema([("x", pa.utf8())])
    batch = pa.record_batch({"x": pa.array(["hello", "world"], type=pa.large_utf8())})
    lying_reader = pa.RecordBatchReader.from_batches(utf8_schema, [batch])
    assert lying_reader.schema.field("x").type == pa.utf8()
    t = con.read_record_batches(lying_reader)
    assert t.execute()["x"].tolist() == ["hello", "world"]


def test_read_record_batches_stream_cache_casts_to_logical_schema() -> None:
    con = xo.connect()
    logical_schema = pa.schema([("x", pa.utf8())])
    batch = pa.record_batch({"x": pa.array(["hello", "world"], type=pa.large_utf8())})
    # Cast before entering StreamCache (as remote_table_exec does) to avoid
    # C Data Interface corruption from large_utf8-vs-utf8 offset mismatch.
    cast_reader = pa.RecordBatchReader.from_batches(
        logical_schema,
        (b.select(logical_schema.names).cast(logical_schema) for b in [batch]),
    )
    cache = StreamCache(cast_reader)
    t = con.read_record_batches(cache, schema=logical_schema)
    assert t.execute()["x"].tolist() == ["hello", "world"]


def test_read_record_batches_stream_cache_schema_drops_extra_columns() -> None:
    con = xo.connect()
    batch = pa.record_batch({"a": [1, 2], "extra": [9, 9]})
    reader = pa.RecordBatchReader.from_batches(batch.schema, [batch])
    cache = StreamCache(reader)
    logical_schema = pa.schema([("a", pa.int64())])
    t = con.read_record_batches(cache, schema=logical_schema)
    result = t.execute()
    assert result.columns.tolist() == ["a"]
    assert result["a"].tolist() == [1, 2]


def test_read_record_batches_from_table() -> None:
    con = xo.connect()
    t = con.read_record_batches(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    result = t.execute()
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == ["x", "y", "z"]


def test_read_record_batches_from_list():
    con = xo.connect()
    batches = [
        pa.record_batch({"a": [1, 2], "b": ["x", "y"]}),
        pa.record_batch({"a": [3], "b": ["z"]}),
    ]
    t = con.read_record_batches(batches)
    result = t.execute()
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == ["x", "y", "z"]


def test_read_record_batches_from_tuple():
    con = xo.connect()
    batches = (
        pa.record_batch({"n": [10, 20]}),
        pa.record_batch({"n": [30]}),
    )
    t = con.read_record_batches(batches)
    assert t.execute()["n"].tolist() == [10, 20, 30]


def test_read_record_batches_from_generator():
    con = xo.connect()

    def gen():
        yield pa.record_batch({"v": [1.0, 2.0]})
        yield pa.record_batch({"v": [3.0]})

    t = con.read_record_batches(gen())
    assert t.execute()["v"].tolist() == [1.0, 2.0, 3.0]


def test_read_record_batches_empty_raises():
    con = xo.connect()
    with pytest.raises(ValueError, match="no rows"):
        con.read_record_batches([])


def test_read_record_batches_empty_table_returns_empty():
    con = xo.connect()
    t = con.read_record_batches(pa.table({"a": pa.array([], type=pa.int64())}))
    assert t.execute().empty


def test_read_record_batches_empty_generator_raises():
    con = xo.connect()

    def empty():
        return
        yield  # noqa: B901

    with pytest.raises(ValueError, match="no rows"):
        con.read_record_batches(empty())


def test_read_record_batches_empty_reader_returns_empty():
    con = xo.connect()
    schema = pa.schema([("a", pa.int64())])
    reader = pa.RecordBatchReader.from_batches(schema, [])
    t = con.read_record_batches(reader)
    assert t.execute().empty


def test_read_record_batches_wrong_type_raises():
    con = xo.connect()
    with pytest.raises(TypeError, match="unsupported source type"):
        con.read_record_batches(42)


def test_read_record_batches_string_raises():
    con = xo.connect()
    with pytest.raises(TypeError, match="unsupported source type"):
        con.read_record_batches("not_a_path")


def test_read_record_batches_schema_mismatch_raises():
    # Passing batches with incompatible schema (field missing) should raise at cast time.
    con = xo.connect()
    first = pa.record_batch({"a": [1, 2], "b": [3, 4]})
    second = pa.record_batch({"a": [5]})  # missing column "b"
    with pytest.raises(ValueError, match="batch schema mismatch"):
        con.read_record_batches([first, second]).execute()


def test_read_record_batches_extra_columns_silently_dropped():
    # Schema inferred from first batch; extra columns in later batches are dropped, not raised.
    con = xo.connect()
    first = pa.record_batch({"a": [1, 2], "b": [3, 4]})
    second = pa.record_batch({"a": [5], "b": [6], "extra": [7]})
    result = con.read_record_batches([first, second]).execute()
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_read_record_batches_reader_extra_columns_silently_dropped():
    # RecordBatchReader path: declared schema lacks "extra"; batch carries it → dropped, not raised.
    con = xo.connect()
    declared = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    batch = pa.record_batch({"a": [1, 2], "b": [3, 4], "extra": [5, 6]})
    reader = pa.RecordBatchReader.from_batches(declared, [batch])
    result = con.read_record_batches(reader).execute()
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_register_unbound_expr_raises():
    con = xo.connect()
    t = xo.table({"a": "int64"}, name="unbound")
    with pytest.raises(ValueError, match="unbound tables"):
        con.register(t)
