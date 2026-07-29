# ADR-0016: Hash identity comes from xorq-owned canonical forms, never dependency-serializer bytes

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
   backend re-execution), per column `combine_chunks` → recursively decode
   dictionary encoding → metadata-free single-column RecordBatch → xxh128
   of its IPC bytes; plus an explicit `(name, str(type))` schema tuple and
   row count.

2. **No normalizer may hash a dependency serializer's byte output** (Arrow
   IPC streams, parquet bytes, pickle) as identity unless the exact byte
   surface is covered by a cross-version stability check. The canonical
   column digest is currently the one sanctioned use: it was verified
   byte-identical across pyarrow 18.0.0–25.0.0 (probe on #2191), and its
   body is isolated so a buffer-level logical fold can replace it without
   another design change.

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
   erased (union / list_view / run-end nesting), refuse loudly rather than
   hash indices without values.

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
  choices) but more per-type code; deferred as the designated escalation:
  swap the body of `canonical_column_digest` and bump
  `NORMALIZATION_VERSION` if a future pyarrow breaks IPC byte stability
  (see #2194's monitor and escalation policy).
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

## References

- #2191 (investigation + cross-version probe script), #2192
  (implementation), #2193 (content-hash read default), #2194
  (version-matrix monitor + provenance guard)
- ADR-0015 (build-hash/cache-hash split; correctness > efficiency)
- Arrow format versioning and columnar spec:
  https://arrow.apache.org/docs/format/Versioning.html,
  https://arrow.apache.org/docs/format/Columnar.html
