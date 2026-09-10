from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from xorq.common.utils.arrow_utils import (
    PANDAS_METADATA_KEY,
    drop_pandas_schema_metadata,
    has_pandas_schema_metadata,
)


@pytest.fixture
def pandas_sourced_table() -> pa.Table:
    return pa.Table.from_pandas(pd.DataFrame({"k": ["x"], "v": [1]}))


def test_from_pandas_carries_the_metadata(pandas_sourced_table: pa.Table) -> None:
    assert has_pandas_schema_metadata(pandas_sourced_table.schema)


def test_drop_table(pandas_sourced_table: pa.Table) -> None:
    dropped = drop_pandas_schema_metadata(pandas_sourced_table)
    assert dropped.schema.metadata is None
    assert dropped.equals(pandas_sourced_table)


def test_drop_batch(pandas_sourced_table: pa.Table) -> None:
    (batch,) = pandas_sourced_table.to_batches()
    dropped = drop_pandas_schema_metadata(batch)
    assert dropped.schema.metadata is None
    assert dropped.to_pydict() == batch.to_pydict()


def test_drop_reader(pandas_sourced_table: pa.Table) -> None:
    dropped = drop_pandas_schema_metadata(pandas_sourced_table.to_reader())
    assert dropped.schema.metadata is None
    assert dropped.read_all().to_pydict() == pandas_sourced_table.to_pydict()


def test_drop_dictionary_reader() -> None:
    """A dictionary column has no self-cast kernel in every Arrow release."""
    table = pa.Table.from_pandas(pd.DataFrame({"c": pd.Categorical(["a", "b"])}))
    assert has_pandas_schema_metadata(table.schema)

    dropped = drop_pandas_schema_metadata(table.to_reader())
    assert dropped.schema.metadata is None
    assert dropped.schema.field("c").type == table.schema.field("c").type
    assert dropped.read_all().to_pydict() == table.to_pydict()


def test_drop_reader_falls_back_when_cast_is_unsupported(
    pandas_sourced_table: pa.Table,
) -> None:
    class NoCastReader:
        """``RecordBatchReader`` is immutable, so stand in for one."""

        schema = pandas_sourced_table.schema

        def __iter__(self):
            return iter(pandas_sourced_table.to_batches())

        def cast(self, schema):
            raise pa.ArrowNotImplementedError("no cast kernel")

    handler = drop_pandas_schema_metadata.registry[pa.RecordBatchReader]
    dropped = handler(NoCastReader())
    assert dropped.schema.metadata is None
    assert dropped.read_all().to_pydict() == pandas_sourced_table.to_pydict()


def test_drop_in_memory_dataset(pandas_sourced_table: pa.Table) -> None:
    dataset = ds.dataset(pandas_sourced_table)
    dropped = drop_pandas_schema_metadata(dataset)
    assert dropped.schema.metadata is None
    assert dropped.to_table().schema.metadata is None
    assert dropped.to_table().to_pydict() == pandas_sourced_table.to_pydict()


def test_drop_filesystem_dataset(pandas_sourced_table: pa.Table, tmp_path) -> None:
    pq.write_table(pandas_sourced_table, tmp_path / "t.parquet")
    dataset = ds.dataset(tmp_path, format="parquet")
    assert has_pandas_schema_metadata(dataset.schema)

    dropped = drop_pandas_schema_metadata(dataset)
    assert dropped.schema.metadata is None
    assert dropped.to_table().schema.metadata is None
    assert dropped.to_table().to_pydict() == pandas_sourced_table.to_pydict()


def test_drop_schema(pandas_sourced_table: pa.Table) -> None:
    dropped = drop_pandas_schema_metadata(pandas_sourced_table.schema)
    assert dropped.metadata is None
    assert dropped == pandas_sourced_table.schema.remove_metadata()


def test_other_schema_metadata_preserved(pandas_sourced_table: pa.Table) -> None:
    table = pandas_sourced_table.replace_schema_metadata(
        {**pandas_sourced_table.schema.metadata, b"mine": b"keep"}
    )
    assert drop_pandas_schema_metadata(table).schema.metadata == {b"mine": b"keep"}


def test_field_metadata_preserved() -> None:
    field = pa.field("k", pa.string(), metadata={b"field": b"keep"})
    table = pa.Table.from_pandas(
        pd.DataFrame({"k": ["x"]}), schema=pa.schema([field])
    ).replace_schema_metadata({PANDAS_METADATA_KEY: b"{}"})
    dropped = drop_pandas_schema_metadata(table)
    assert dropped.schema.metadata is None
    assert dropped.schema.field("k").metadata == {b"field": b"keep"}


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="no-metadata"),
        pytest.param({b"mine": b"keep"}, id="other-metadata"),
    ],
)
def test_without_the_key_is_identity(
    pandas_sourced_table: pa.Table, metadata: dict[bytes, bytes] | None
) -> None:
    table = pandas_sourced_table.replace_schema_metadata(metadata)
    assert drop_pandas_schema_metadata(table) is table


def test_unsupported_type_raises() -> None:
    with pytest.raises(TypeError, match="Cannot drop pandas schema metadata"):
        drop_pandas_schema_metadata(pd.DataFrame({"k": ["x"]}))
