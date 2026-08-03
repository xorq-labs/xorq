"""Canonical, pyarrow-version-stable hashing of in-memory table data.

Replaces xorq_dasher's ``normalize_inmemorytable`` /
``normalize_memory_databasetable`` / ``normalize_pyarrow_table``, which hash
the IPC bytes of a backend-executed ``to_pyarrow_batches()`` stream. That
stream is an incidental physical artifact: its bytes depend on batch size
and on pyarrow's IPC dictionary-batch handling, which changed between
pyarrow 20 and 21 and silently renamed every memtable-bearing build
(issue #2191).

The canonical form hashed here is a *logical* one:

* never re-execute an ``InMemoryTable`` through a backend — hash the stored
  proxy data (``op.data.to_pyarrow(op.schema)``);
* per column: recursively decode dictionary encoding, widen int32
  offsets to ``large_*``, and rewrite ``string_view``/``binary_view`` to
  their ``large_*`` offset forms (all erase physical encoding; an
  un-decoded dictionary batch message would serialize indices *without*
  the dictionary values — an under-discrimination hazard — and a view
  array serializes its physical buffer layout, which varies with how the
  array was built), compact via ``combine_chunks``/``concat_arrays``
  (erases chunk layout and slicing), then xxh128 the IPC bytes of a
  metadata-free single-column RecordBatch;
* carry the schema as an explicit ``(name, str(canonical type))`` tuple —
  ``RecordBatch.serialize()`` emits no schema message, so type identity
  must not be left to the value bytes alone;
* refuse what cannot be canonicalized (extension types, list-view types,
  dictionaries nested where they cannot be decoded away) with
  ``NotImplementedError`` rather than hash them unstably or collidingly.

The column digest was verified byte-identical across pyarrow 18.0.0, 20.0.0,
21.0.0 and 25.0.0 for a corpus covering nulls, NaN/-0.0/inf, decimals, tz
timestamps, nested list/struct/map, multi-chunk dictionary arrays,
string_view/binary_view arrays (plain, sliced and chunked), and
sliced/chunked var-length arrays — rerun
``scripts/canonical_digest_xver_probe.py`` (a standalone replica of this
logic) to re-verify. If a future pyarrow breaks IPC byte stability, swap the
body of :func:`canonical_column_digest` for a buffer-level logical fold and
bump ``NORMALIZATION_VERSION``.

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


def _assert_canonicalizable(typ: pa.DataType) -> None:
    """Refuse types whose logical identity this module cannot provably
    capture, walking child fields generically (so an extension type nested
    in any family — list, struct, map, union, list_view, … — is found).

    Extension types are refused outright: their parameters live in
    ``__arrow_ext_serialize__()`` bytes — third-party serialization with no
    cross-version stability contract (hashing them would re-introduce the
    dependency-drift coupling of #2191) — while ``str(type)`` omits them
    entirely, so passing the storage through would collide logically
    different tables (e.g. pandas period columns of freq 'M' vs 'D' with
    equal ordinals). Loud refusal is the only option that neither collides
    nor version-couples; support requires a ``NORMALIZATION_VERSION`` bump.

    List-view types are refused because they cannot be canonicalized:
    pyarrow's ``list_view`` → ``list`` cast emits invalid arrays (offsets
    fail ``validate(full=True)``; verified on pyarrow 18 and 21), and
    passing a view array through uncanonicalized would serialize its
    physical buffer layout — equal data hashing unequally, breaking the
    layout-invariance contract. (``string_view``/``binary_view`` cast
    correctly and are canonicalized by :func:`_canonical_type` instead.)

    Union and run-end-encoded types are refused for the same
    layout-invariance reason: a sparse union serializes its type-masked
    child slots (logically invisible values enter the bytes) and a
    run-end-encoded array serializes its run partitioning (``[3,5]/[7,9]``
    vs ``[2,3,5]/[7,7,9]`` encode the same values), so equal data hashes
    unequally, and neither has a verified canonicalizing cast. Support for
    either requires a decode step plus a ``NORMALIZATION_VERSION`` bump.

    The child walk cannot rely on ``num_fields`` alone:
    ``pa.DictionaryType.num_fields == 0``, so a dictionary's value type
    must be descended explicitly — an extension type smuggled in as a
    dictionary's value type would otherwise slip past this refusal and
    collide (found by the round-2 cold review).
    """
    import pyarrow as pa  # noqa: PLC0415

    if isinstance(typ, pa.BaseExtensionType):
        raise NotImplementedError(
            f"cannot canonically hash extension type {typ!r}: its parameters "
            f"(__arrow_ext_serialize__ bytes) have no cross-version stability "
            f"contract and str(type) omits them, so logically different "
            f"tables would collide"
        )
    if pa.types.is_list_view(typ) or pa.types.is_large_list_view(typ):
        raise NotImplementedError(
            f"cannot canonically hash list-view type {typ!r}: pyarrow's "
            f"list_view->list cast produces invalid arrays, and serializing "
            f"the view uncanonicalized would hash its physical buffer layout "
            f"instead of its logical values"
        )
    if pa.types.is_union(typ):
        raise NotImplementedError(
            f"cannot canonically hash union type {typ!r}: a sparse union "
            f"serializes its type-masked child slots, so equal data can "
            f"hash unequally"
        )
    if pa.types.is_run_end_encoded(typ):
        raise NotImplementedError(
            f"cannot canonically hash run-end-encoded type {typ!r}: its "
            f"serialized form depends on the run partitioning, so equal "
            f"data can hash unequally"
        )
    if pa.types.is_dictionary(typ):
        _assert_canonicalizable(typ.value_type)
    for i in range(typ.num_fields):
        _assert_canonicalizable(typ.field(i).type)


def _canonical_type(typ: pa.DataType) -> pa.DataType:
    """Return the canonical logical type for ``typ``: every dictionary layer
    (at any nesting depth) replaced by its logical value type, and
    int32-offset and view var-length types rewritten to their ``large_*``
    (int64-offset) variants.

    Widening does double duty: offset width is a physical encoding, not a
    logical property (ibis maps string and large_string to the same dtype),
    and a single contiguous int32-offset array can address at most 2 GiB of
    value data — a chunked column can hold more in memory just fine, so
    canonicalizing to one contiguous array would overflow where the widened
    type cannot. View types additionally MUST be rewritten because their
    IPC form serializes the physical buffer layout — equal data built
    differently serializes differently. (``map`` has no large variant in
    Arrow; a >2^31-entry map column fails loudly in ``combine_chunks``
    rather than mis-hashing. Likewise a ``dictionary`` of view values —
    pyarrow cannot decode it, so the cast fails loudly.)
    """
    import pyarrow as pa  # noqa: PLC0415

    if pa.types.is_dictionary(typ):
        return _canonical_type(typ.value_type)
    if pa.types.is_string(typ) or pa.types.is_string_view(typ):
        return pa.large_string()
    if pa.types.is_binary(typ) or pa.types.is_binary_view(typ):
        return pa.large_binary()
    if pa.types.is_list(typ) or pa.types.is_large_list(typ):
        return pa.large_list(_canonical_type(typ.value_type))
    if pa.types.is_fixed_size_list(typ):
        return pa.list_(_canonical_type(typ.value_type), typ.list_size)
    if pa.types.is_struct(typ):
        return pa.struct(
            [
                (typ.field(i).name, _canonical_type(typ.field(i).type))
                for i in range(typ.num_fields)
            ]
        )
    if pa.types.is_map(typ):
        return pa.map_(
            _canonical_type(typ.key_type),
            _canonical_type(typ.item_type),
            keys_sorted=typ.keys_sorted,
        )
    return typ


def _contains_dictionary(typ: pa.DataType) -> bool:
    """True if ``typ`` has a dictionary layer anywhere in its nesting.

    Backstop, believed unreachable for currently-supported families: every
    field-bearing family :func:`_canonical_type` does not rewrite (union,
    run-end encoded, list-view) is refused in
    :func:`_assert_canonicalizable` before this runs. It stays because it
    is cheap and a future pyarrow type family could reopen the path — an
    un-erased dictionary hashing indices without values is the worst
    silent failure this module can have."""
    import pyarrow as pa  # noqa: PLC0415

    if pa.types.is_dictionary(typ):
        return True
    return any(_contains_dictionary(typ.field(i).type) for i in range(typ.num_fields))


def canonical_column_digest(col: pa.Array | pa.ChunkedArray) -> str:
    """xxh128 digest of one column's logical values, physical-layout-free.

    Identical for chunked vs contiguous, sliced vs compact,
    dictionary-encoded vs plain, and int32- vs int64-offset representations
    of the same values (invariants asserted in ``test_hash_contract.py``).

    The digest carries NO type identity: ``RecordBatch.serialize()`` emits
    no schema message, so e.g. int8 and uint8 arrays with identical buffers
    produce identical digests. Never use it as a standalone identity — it
    must always be paired with the column's logical type, as
    :func:`normalize_pyarrow_table_canonical` does via its schema component.
    """
    import pyarrow as pa  # noqa: PLC0415

    _assert_canonicalizable(col.type)
    canonical_type = _canonical_type(col.type)
    if _contains_dictionary(canonical_type):
        # A dictionary layer survived canonicalization — only possible for
        # a type family this module does not know (the known un-rewritable
        # families are refused in _assert_canonicalizable). Serializing it
        # would emit indices *without* the dictionary values — refuse
        # loudly rather than under-discriminate.
        raise NotImplementedError(
            f"cannot erase dictionary encoding nested in {canonical_type!r}; "
            f"hashing it would drop the dictionary values from the digest"
        )
    if canonical_type != col.type:
        # ChunkedArray.cast casts chunk-wise, so offset widening happens
        # BEFORE the chunks are combined — a >2 GiB string column would
        # otherwise overflow its int32 offsets in combine_chunks.
        col = col.cast(canonical_type)
    # Both branches funnel through concat_arrays, which rebuilds buffers
    # compactly: a slice of a var-length array carries un-rebased offsets
    # that serialize differently from a fresh equivalent. The per-type
    # slicing invariants in test_hash_contract.py pin this compaction
    # behavior — if a future pyarrow adds a zero-copy fast path, they fail
    # there, not in production.
    if isinstance(col, pa.ChunkedArray):
        arr = col.combine_chunks()
    else:
        arr = pa.concat_arrays([col])
    batch = pa.RecordBatch.from_arrays([arr], names=["c"])
    return xxhash.xxh128(batch.serialize().to_pybytes()).hexdigest()


def normalize_pyarrow_table_canonical(table: pa.Table) -> tuple:
    """Canonical normalization of a ``pa.Table``: row count + logical schema
    + per-column value digests. Schema/field metadata (where ``from_pandas``
    embeds pandas/pyarrow version strings) is deliberately excluded. The row
    count is carried explicitly because a zero-column table has no digests
    to carry it implicitly. The schema carries the *canonical* type string,
    so representations that differ only in physical encoding (dictionary
    vs plain, string vs large_string) normalize identically.

    Deliberately excluded from identity, at every nesting level: field
    nullability (a constraint on future writes, not data —
    ``_canonical_type``'s reconstruction drops nested ``not null`` and the
    tuple below omits ``field.nullable``) and the dictionary ``ordered``
    flag (meaningless once the encoding it qualifies is erased); both
    exclusions are pinned in ``test_hash_contract.py``."""
    for field in table.schema:
        _assert_canonicalizable(field.type)
    return (
        "xorq.pa.Table",
        NORMALIZATION_VERSION,
        table.num_rows,
        tuple((field.name, str(_canonical_type(field.type))) for field in table.schema),
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
    introduces. Streaming ``to_pyarrow_batches()`` here would save no peak
    memory: the canonical digest compacts each column across ALL chunks
    (that compaction is what erases batch boundaries), so every batch must
    be resident anyway — and an incremental per-batch fold would re-couple
    the hash to batch boundaries, the exact defect being removed."""
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
