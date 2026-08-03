# ADR-0017: Hash identity comes from xorq-owned canonical forms, never dependency-serializer bytes

- **Status:** Proposed
- **Date:** 2026-07-29
- **Deciders:** Dan Lovell

## Context

Content-addressed identity (build hashes, cache keys, deterministic memtable
names) was partly derived from the bytes that dependency serializers happen
to produce: `xorq_dasher.normalize_inmemorytable` executed a memtable
through a backend and xxhashed the IPC bytes of the resulting
`to_pyarrow_batches()` stream, and `normalize_pyarrow_table` hashed
serialized batch bytes directly.

Arrow's format versioning guarantees *logical readability* across versions,
not byte determinism — the columnar spec leaves padding bytes, null-slot
contents, and validity-bitmap presence unspecified even within one version.
When pyarrow changed its IPC dictionary-batch handling between 20 and 21,
every memtable-bearing build silently got a new name: `ci-test-lowest-direct`
broke when uv's resolver drifted pyarrow 21 → 20 (#2191), and the same
pipeline built on adjacent pyarrow versions cache-missed against itself.

The same failure family (hashing an incidental physical/environment artifact
as if it were logical identity) recurs elsewhere: parquet bytes embed
`created_by` version strings and change encoding defaults at 19→20 and
21→22; cloudpickle output is Python-version-coupled; generated SQL text is
sqlglot-coupled.

## Decision drivers

- **Correctness over efficiency (ADR-0015):** under-discrimination
  (different data → same hash → wrong result) is never acceptable;
  over-discrimination (same data → different hash → recompute) is a cost to
  drive down.
- Hash identity must change when *xorq decides*, not when a dependency
  releases.
- Reproducible builds: the same expression must produce the same artifact
  name on any supported dependency matrix.
- No fork of `xorq_dasher`: xorq already layers rule overrides via
  `DEFAULT_HASHER.override(...)`.

## Decision

1. **Identity is a hash of a logical canonical form, defined and versioned
   by xorq** (`python/xorq/common/utils/dasher/_canonical.py`). For
   in-memory table data the canonical form is: stored proxy data (never a
   backend re-execution), then per column — recursively decode dictionary
   encoding, widen int32-offset var-length types to `large_*`, and rewrite
   `string_view`/`binary_view` to `large_string`/`large_binary` (one
   chunk-wise `cast`) → `combine_chunks`/`concat_arrays` → metadata-free
   single-column RecordBatch → xxh128 of its IPC bytes; plus an explicit
   `(name, str(canonical type))` schema tuple and row count.

   The cast **precedes** compaction and the order is load-bearing: a
   >2 GiB string column holds more value data than one contiguous
   int32-offset array can address, so compacting before widening would
   overflow.

2. **No normalizer may hash or embed a dependency-owned representation**
   (Arrow IPC streams, parquet bytes, pickle, type reprs) as identity
   unless that exact surface is covered by a cross-version stability
   check. Two surfaces are currently sanctioned, both verified across
   pyarrow 18.0.0–25.0.0 by `scripts/canonical_digest_xver_probe.py`
   (#2191) and both emitted by that probe so the check keeps running:

   - the canonical column digest (IPC bytes), whose body is isolated so a
     buffer-level logical fold can replace it without another design
     change; and
   - `str(canonical type)` in the schema component, i.e. pyarrow's
     `DataType.__str__`. It carries real identity — `RecordBatch.serialize()`
     emits no schema message, so int8 and uint8 with equal buffers are
     otherwise indistinguishable — and is therefore as much a stability
     dependency as the digest, not a free-form label.

3. **Every canonical tuple embeds `NORMALIZATION_VERSION`.** Changing any
   canonical form requires bumping it — a deliberate, release-noted,
   fleet-wide cache-invalidation event — and regenerating the golden tokens.

4. **Golden-token contract tests**
   (`python/xorq/common/utils/tests/test_hash_contract.py`) pin the token
   output per surface. Any change to token output fails in the PR that
   causes it, with a message carrying dependency versions and the
   normalized tuple for diagnosis.

5. **Canonicalization must be audited for omissions.** A serializer's
   omissions are under-discrimination holes: `RecordBatch.serialize()`
   carries no schema message (int8 vs uint8 with identical buffers collide
   without an explicit type component) and no dictionary values (encoded
   indices hash alone unless decoded). Where a dictionary layer cannot be
   erased, refuse loudly rather than hash indices without values — and the
   walk that looks for such types must descend a dictionary's value type
   explicitly, since `pa.DictionaryType.num_fields == 0`.

   Separately and unconditionally, four families are refused outright
   because no verified canonicalizing cast exists for them: extension types
   (parameters live in `__arrow_ext_serialize__` bytes that `str(type)`
   omits — an under-discrimination hole), and list-view, union and
   run-end-encoded types (their serialized form is unboundedly
   layout-coupled). This is independent of whether they nest a dictionary.

## Alternatives considered

- **Pin pyarrow to a proven-stable block (`>=21,<22`).** Rejected as the
  fix: it narrows the supported range, must be re-litigated every pyarrow
  release, and hides rather than removes the coupling. (It was also skipped
  as a stopgap once the canonical digest was verified stable on 18–25.)
- **Scrub version strings from serialized bytes.** Insufficient: the pa20↔21
  divergence is a real dictionary-batch layout change, not a metadata
  string; parquet 21→22 changes live inside SNAPPY-compressed bytes.
- **Fully logical buffer-level hash (per-type folds over value bytes, null
  masks, offsets).** Strictly more robust (immune to IPC framing/padding
  choices, and the only thing that erases the null-slot-payload residual
  below at every nesting level) but more per-type code; deferred as the
  designated escalation: swap the body of `canonical_column_digest` and bump
  `NORMALIZATION_VERSION` if a future pyarrow breaks IPC byte stability (see
  #2194's monitor and escalation policy). Rejected as a partial measure:
  zeroing null payloads for top-level primitives only would leave nested
  children (struct/list of nullable primitives) untouched, reproducing the
  non-uniform-exclusion defect that round 3 fixed for `keys_sorted`.
- **Upstream a canonical-bytes mode into Arrow.** Arrow explicitly declines
  byte-stability guarantees; not a realistic dependency.

## Consequences

- One-time hash bump for memtable-bearing expressions (#2192); thereafter
  artifact names are stable across the supported dependency matrix.
- Deterministic memtable naming no longer executes data through a backend
  (also removes a datafusion round-trip from tokenization).
- New rules that hash serializer bytes must either canonicalize or join the
  cross-version monitor (#2194 adds an allowlist guard test enforcing
  this).
- Known over-discrimination surfaces remain (UDF cloudpickle tokens,
  generated SQL text) — acceptable per ADR-0015's asymmetry, tracked by the
  monitor rather than golden tokens.
- Two over-discrimination residuals are accepted *inside* the canonical form
  itself, priced as recomputation and pinned by tests in
  `test_hash_contract.py` rather than left undocumented:
  - **Null-slot payload.** The columnar spec leaves null-slot contents
    unspecified and `RecordBatch.serialize()` writes the values buffer
    verbatim, so producers disagree on `.equals()`-identical data:
    `pa.Table.from_pandas` leaves NaN/NaT/mask-garbage under a null where a
    literal `None` leaves zero (float64, nullable `Int64`, `datetime64`
    confirmed). Not refused, unlike union/run-end-encoded — those have
    unbounded, non-producer-stable layout variance and are rare, whereas
    refusing null-slot variance would refuse nearly every nullable column.
    Closing it is exactly the deferred buffer-level fold below.
  - **Timezone spelling.** `str(type)` distinguishes `tz=UTC` from
    `tz=+00:00` for the same instant; the value digests agree.

  Neither re-opens #2191: both are deterministic given the producer, so
  identity never moves because a dependency moved.

## References

- #2191 (investigation + cross-version probe script), #2192
  (implementation), #2193 (content-hash read default), #2194
  (version-matrix monitor + provenance guard)
- ADR-0015 (build-hash/cache-hash split; correctness > efficiency)
- Arrow format versioning and columnar spec:
  https://arrow.apache.org/docs/format/Versioning.html,
  https://arrow.apache.org/docs/format/Columnar.html
