import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo


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


def test_read_record_batches_type_mismatch():
    # register_and_transform_remote_tables builds a RecordBatchReader with
    # schema from ex.as_table().schema().to_pyarrow() (ibis-declared, e.g.
    # utf8 for VARCHAR) but batches from ex.to_pyarrow_batches() which may
    # emit a different physical type (e.g. large_utf8 for DuckDB VARCHAR).
    # Passing this "lying reader" directly to DataFusion via C Data Interface
    # silently corrupts data: the utf8 offset buffer is 32-bit but large_utf8
    # data uses 64-bit offsets, so DataFusion misreads row boundaries.
    # read_record_batches must cast each batch to the declared schema before
    # handing the reader to DataFusion.
    con = xo.connect()
    utf8_schema = pa.schema([("x", pa.utf8())])
    batch = pa.record_batch({"x": pa.array(["hello", "world"], type=pa.large_utf8())})
    # RecordBatchReader.from_batches does not validate batch types against the
    # declared schema, reproducing the lying reader from production code paths.
    lying_reader = pa.RecordBatchReader.from_batches(utf8_schema, [batch])
    assert lying_reader.schema.field("x").type == pa.utf8()

    # Naive registration (bypassing read_record_batches) silently corrupts:
    con.con.register_record_batch_reader("naive_table", lying_reader)
    naive_result = con.table("naive_table").execute()["x"].tolist()
    assert naive_result != ["hello", "world"], "naive registration should corrupt data"

    # read_record_batches casts each batch before crossing the C boundary:
    fresh_reader = pa.RecordBatchReader.from_batches(
        utf8_schema,
        [pa.record_batch({"x": pa.array(["hello", "world"], type=pa.large_utf8())})],
    )
    t = con.read_record_batches(fresh_reader)
    assert t.execute()["x"].tolist() == ["hello", "world"]


def test_read_record_batches_from_table():
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
    with pytest.raises(ValueError, match="no record batches"):
        con.read_record_batches([])


def test_read_record_batches_empty_generator_raises():
    con = xo.connect()

    def empty():
        return
        yield  # noqa: B901

    with pytest.raises(ValueError, match="no record batches"):
        con.read_record_batches(empty())


def test_read_record_batches_wrong_type_raises():
    con = xo.connect()
    with pytest.raises((TypeError, AttributeError)):
        con.read_record_batches("not_a_source")


def test_read_record_batches_schema_mismatch_raises():
    # Passing batches with incompatible schema (field missing) should raise at cast time.
    con = xo.connect()
    first = pa.record_batch({"a": [1, 2], "b": [3, 4]})
    second = pa.record_batch({"a": [5]})  # missing column "b"
    with pytest.raises(ValueError, match="does not exist in schema"):
        con.read_record_batches([first, second]).execute()


def test_register_unbound_expr_raises():
    con = xo.connect()
    t = xo.table({"a": "int64"}, name="unbound")
    with pytest.raises(ValueError, match="unbound tables"):
        con.register(t)
