"""Helpers for normalizing Arrow data before handing it to an engine."""

from __future__ import annotations

import functools

import pyarrow as pa
import pyarrow.dataset as ds


PANDAS_METADATA_KEY = b"pandas"


def has_pandas_schema_metadata(schema: pa.Schema) -> bool:
    return PANDAS_METADATA_KEY in (schema.metadata or {})


def _drop_key(metadata: dict[bytes, bytes] | None) -> dict[bytes, bytes] | None:
    rest = {k: v for k, v in (metadata or {}).items() if k != PANDAS_METADATA_KEY}
    return rest or None


@functools.singledispatch
def drop_pandas_schema_metadata(obj: object) -> object:
    """Drop the ``pandas`` schema-level metadata pyarrow attaches on conversion.

    DataFusion compares schemas *including* schema-level metadata in the
    ``join_selection`` physical optimizer rule and in its logical/physical
    schema verifier, so joining two tables that carry different
    ``{"pandas": ...}`` blobs fails with an internal "Schema mismatch" error
    (xorq #2266). The blob is descriptive only as far as xorq is concerned:
    index columns are materialized or dropped before registration, and
    memtable hashing normalizes schema metadata away already (see
    ``xorq.common.utils.dasher._canonical``).

    Other metadata keys and all field-level metadata (Arrow extension types)
    are preserved. Objects without the key are returned unchanged.
    """
    raise TypeError(f"Cannot drop pandas schema metadata from {type(obj)}")


@drop_pandas_schema_metadata.register(pa.Schema)
def _schema(obj: pa.Schema) -> pa.Schema:
    if not has_pandas_schema_metadata(obj):
        return obj
    rest = _drop_key(obj.metadata)
    return obj.remove_metadata() if rest is None else obj.with_metadata(rest)


@drop_pandas_schema_metadata.register(pa.Table)
@drop_pandas_schema_metadata.register(pa.RecordBatch)
def _table_or_batch(obj: pa.Table | pa.RecordBatch) -> pa.Table | pa.RecordBatch:
    if not has_pandas_schema_metadata(obj.schema):
        return obj
    return obj.replace_schema_metadata(_drop_key(obj.schema.metadata))


@drop_pandas_schema_metadata.register(pa.RecordBatchReader)
def _reader(obj: pa.RecordBatchReader) -> pa.RecordBatchReader:
    """Wrap the reader, restating its schema and each batch's without the blob.

    A reader cannot be rewritten in place, so a carrying reader comes back as a
    new one draining the original. The original is then owned by the wrapper:
    errors it raises surface from the wrapper as it is consumed, and discarding
    the wrapper without consuming it leaves the original unread.
    """
    if not has_pandas_schema_metadata(obj.schema):
        return obj
    return pa.RecordBatchReader.from_batches(
        drop_pandas_schema_metadata(obj.schema),
        map(drop_pandas_schema_metadata, obj),
    )


@drop_pandas_schema_metadata.register(ds.Dataset)
def _dataset(obj: ds.Dataset) -> ds.Dataset:
    if not has_pandas_schema_metadata(obj.schema):
        return obj
    # replace_schema keeps the fragments and the laziness: only the declared
    # schema changes, so a FileSystemDataset is not read here.
    return obj.replace_schema(drop_pandas_schema_metadata(obj.schema))
