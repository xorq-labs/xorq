"""Canonical, pyarrow-version-stable hashing of in-memory table data.

Replaces xorq_dasher 0.1.0's ``normalize_inmemorytable`` /
``normalize_memory_databasetable`` / ``normalize_pyarrow_table``, which hash
the IPC bytes of a backend-executed ``to_pyarrow_batches()`` stream. That
stream is an incidental physical artifact: its bytes depend on batch size
and on pyarrow's IPC dictionary-batch handling, which changed between
pyarrow 20 and 21 and silently renamed every memtable-bearing build
(issue #2191).

The canonical form hashed here is a *logical* one:

* never re-execute an ``InMemoryTable`` through a backend — hash the stored
  proxy data (``op.data.to_pyarrow(op.schema)``);
* per column: ``combine_chunks`` (erases chunk layout), recursively decode
  dictionary encoding (erases physical encoding; an un-decoded dictionary
  batch message would serialize indices *without* the dictionary values —
  an under-discrimination hazard), then xxh128 the IPC bytes of a
  metadata-free single-column RecordBatch;
* carry the schema as an explicit ``(name, str(type))`` tuple —
  ``RecordBatch.serialize()`` emits no schema message, so type identity
  must not be left to the value bytes alone.

The column digest was verified byte-identical across pyarrow 18.0.0–25.0.0
for a corpus covering nulls, NaN/-0.0/inf, decimals, tz timestamps, nested
list/struct/map, multi-chunk dictionary arrays, and sliced arrays (probe
script on issue #2191). If a future pyarrow breaks IPC byte stability, swap
the body of :func:`canonical_column_digest` for a buffer-level logical fold
and bump ``NORMALIZATION_VERSION``.

``NORMALIZATION_VERSION`` is folded into every tuple produced here: hash
identity is versioned by xorq, deliberately, not by dependency drift. Any
change to the canonical form must bump it (one fleet-wide cache
invalidation) and update the golden tokens in
``python/xorq/common/utils/tests/test_hash_contract.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xxhash

from xorq.common.utils.dasher._gap_rules import normalize_ibis_schema


if TYPE_CHECKING:
    import pyarrow as pa

    from xorq.vendor.ibis.expr.operations.relations import (
        DatabaseTable,
        InMemoryTable,
    )


NORMALIZATION_VERSION = 1


def _dictionary_free_type(typ: pa.DataType) -> pa.DataType:
    """Return ``typ`` with every dictionary layer (at any nesting depth)
    replaced by its logical value type."""
    import pyarrow as pa  # noqa: PLC0415

    if pa.types.is_dictionary(typ):
        return _dictionary_free_type(typ.value_type)
    if pa.types.is_list(typ):
        return pa.list_(_dictionary_free_type(typ.value_type))
    if pa.types.is_large_list(typ):
        return pa.large_list(_dictionary_free_type(typ.value_type))
    if pa.types.is_fixed_size_list(typ):
        return pa.list_(_dictionary_free_type(typ.value_type), typ.list_size)
    if pa.types.is_struct(typ):
        return pa.struct(
            [
                (typ.field(i).name, _dictionary_free_type(typ.field(i).type))
                for i in range(typ.num_fields)
            ]
        )
    if pa.types.is_map(typ):
        return pa.map_(
            _dictionary_free_type(typ.key_type),
            _dictionary_free_type(typ.item_type),
            keys_sorted=typ.keys_sorted,
        )
    return typ


def canonical_column_digest(col: pa.Array | pa.ChunkedArray) -> str:
    """xxh128 digest of one column's logical values, physical-layout-free.

    Identical for chunked vs contiguous, sliced vs compact, and
    dictionary-encoded vs plain representations of the same values
    (invariants asserted in ``test_hash_contract.py``).
    """
    import pyarrow as pa  # noqa: PLC0415

    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    canonical_type = _dictionary_free_type(arr.type)
    if canonical_type != arr.type:
        arr = arr.cast(canonical_type)
    batch = pa.RecordBatch.from_arrays([arr], names=["c"])
    return xxhash.xxh128(batch.serialize().to_pybytes()).hexdigest()


def normalize_pyarrow_table_canonical(table: pa.Table) -> tuple:
    """Canonical normalization of a ``pa.Table``: logical schema + per-column
    value digests. Schema/field metadata (where ``from_pandas`` embeds
    pandas/pyarrow version strings) is deliberately excluded."""
    return (
        "xorq.pa.Table",
        NORMALIZATION_VERSION,
        tuple(
            (field.name, str(_dictionary_free_type(field.type)))
            for field in table.schema
        ),
        tuple(canonical_column_digest(col) for col in table.columns),
    )


def normalize_inmemorytable_canonical(op: InMemoryTable) -> tuple:
    """InMemoryTable identity from its STORED proxy data — no backend
    round-trip, so no dependence on execution batching or a backend's
    choice of physical encoding."""
    return (
        "xorq.InMemoryTable",
        NORMALIZATION_VERSION,
        normalize_ibis_schema(op.schema),
        normalize_pyarrow_table_canonical(op.data.to_pyarrow(op.schema)),
    )


def normalize_memory_databasetable_canonical(dt: DatabaseTable) -> tuple:
    """Memory-backed DatabaseTable identity. The data lives inside the
    backend, so one execution is unavoidable — but hashing the canonical
    form of the materialized table (rather than the raw batch stream)
    erases the batch-size and IPC-dictionary variance that execution
    introduces."""
    return (
        "xorq.MemoryDatabaseTable",
        NORMALIZATION_VERSION,
        normalize_ibis_schema(dt.schema),
        normalize_pyarrow_table_canonical(dt.to_expr().to_pyarrow()),
    )


__all__ = [
    "NORMALIZATION_VERSION",
    "canonical_column_digest",
    "normalize_inmemorytable_canonical",
    "normalize_memory_databasetable_canonical",
    "normalize_pyarrow_table_canonical",
]
