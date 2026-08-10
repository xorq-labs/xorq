# ADR-XXXX: Read-kwargs identity is a deny-list — a read kwarg is identity-bearing unless declared transport

- **Status:** Proposed
- **Date:** 2026-08-07
- **Deciders:** Dan Lovell

> **Proposed, not implemented.** `READ_NON_IDENTITY_KEYS` and
> `READ_TRANSPORT_KEYS` do not exist in the tree; `READ_IDENTITY_KEYS` is still
> the allow-list described below. The defect this ADR responds to is live and
> reproduced — see [#2206](https://github.com/xorq-labs/xorq/issues/2206).

## Context

A path-ful `Read` folds four read kwargs into identity and drops the rest:

```python
# python/xorq/common/constants.py, introduced by #2070 (relocatable reads)
READ_IDENTITY_KEYS = frozenset({"mode", "schema", "temporary", "relocatable"})
```

Two consumers apply it: `_read_extra_kwargs`
(`common/utils/dasher/_relations.py`, build and mtime tokenization) and
`snapshot_normalize_read` (`caching/strategy.py`, snapshot cache keys). The path
itself contributes only content or stat metadata, and the remote branch only
HTTP/object metadata, so a parse kwarg enters identity nowhere.

That is an *allow-list*, and the failure direction of an allow-list is
under-hashing. Reproduced against duckdb — three `deferred_read_csv` reads of
one file differing only in `skip`, returning three, two and one rows, all
producing a single token, and the cached result for the first served for the
others under both `ParquetCache` and `ParquetSnapshotCache`:

```python
a = xo.deferred_read_csv(p, con, schema=sch)          # 3 rows
c = xo.deferred_read_csv(p, con, schema=sch, skip=2)  # 1 row
assert tokenize(a.op()) == tokenize(c.op())           # collide
```

This is exactly the violation ADR-0015 names: something that changes what data
comes back does not change the hash. It is not a REST defect — it predates this
stack and lives in the oldest read path.

Two facts bound it, and both belong in the record:

- **The declared schema is safe.** It is folded via the op's `.schema` on every
  branch, not via `read_kwargs`, so two reads declaring different schemas do not
  collide. The exposure is parse kwargs that change rows *without* changing the
  declared schema.
- **The exploitable population is narrower than the collision.** It requires a
  backend whose read method both accepts and applies a rows-changing kwarg.
  duckdb parse kwargs are the confirmed case; on the default backend `skip` is
  rejected at execution, so tokens still collide but no wrong rows are served.
  A collision that is currently unexploitable on one backend is still a defect:
  it is one accepted kwarg away from being exploitable there too.

The correct shape has been arrived at twice already, both times more recently
than the allow-list:

- `normalize_read_source_identity` (`common/utils/file_utils.py`,
  ADR-api-relations-are-pathless-read-ops) is in the tree today, and folds
  **all** read kwargs minus `table_name` for path-less reads.
- `identity_field_names` (`backends/rest/config.py`,
  ADR-rest-config-contract-identity-folded-residence-either) lands with the REST
  config contract, and derives identity membership from the attrs declaration so
  a new field is identity-bearing by default. It records why: "a hand-written
  tuple beside a growing class drifts silently in the dangerous direction."

So the path-ful reads are the last place where the old default survives.

## Decision drivers

- ADR-0015: anything that changes what data comes back must change the build
  hash. An allow-list cannot honor that for kwargs nobody remembered to list.
- The failure *direction* is the whole argument. Over-hashing costs a recompute;
  under-hashing serves wrong rows. Defaults must fail toward the recompute.
- ADR-0006's properties must survive: same content at a different absolute path
  still tokenizes the same, and location must stay out of identity.
- The fix must cover both consumers. Fixing tokenization alone would leave
  snapshot cache keys colliding.

## Decision

**Invert the allow-list. Every read kwarg is identity-bearing unless it appears
in an explicit deny-list.**

```python
READ_NON_IDENTITY_KEYS = frozenset({"hash_path", "read_path", "table_name"}) | READ_TRANSPORT_KEYS
```

Each exclusion carries its own justification, and there are only three standing
ones:

- `hash_path` — path identity is contributed as content/stat by the path
  branches; folding the raw string as well would break ADR-0006's
  same-content-different-path property.
- `read_path` — location, not identity (ADR-0006).
- `table_name` — `gen_name`'d per construction, so folding it makes a read's
  hash unstable across constructions of the same logical read. This is the same
  exclusion `normalize_read_source_identity` already makes.

`READ_TRANSPORT_KEYS` is the extension point for genuinely transport-only kwargs
— things that change how bytes arrive but not which rows result. It starts
empty or near-empty, and **its exact membership is pinned by a test**, on the
ADR-0016 tripwire pattern. That pin is the load-bearing part: adding a name to
this set is the one edit in the scheme whose failure direction is
non-conservative, so it must move a pinned literal and carry a justification
rather than being a quiet dictionary edit.

Both consumers change together: `_read_extra_kwargs` and
`snapshot_normalize_read` both filter on `not in READ_NON_IDENTITY_KEYS`,
preserving `read_kwargs` order and the existing tuple shape so the emitted
identity is byte-identical for any read whose kwargs were already fully covered.

## Alternatives considered

### Extend the allow-list with the kwargs we know about

Rejected because:
- It repeats the mechanism that produced the defect. The allow-list was not
  wrong about the four names it holds; it was wrong to be a list. Every future
  backend kwarg re-runs the same coin flip, and the losing side is a wrong
  answer rather than a slow one.

### Per-backend declarations of which kwargs are identity-bearing

Each backend declares its own read-kwarg identity set, near the read method
that consumes them.

Deferred because:
- It is more precise and it is where the knowledge actually lives, but it is
  also more surface for the same failure — a backend that declares nothing
  degrades silently to the old behavior. It composes with this ADR later as a
  *narrowing* of an already-safe default; adopting it first would mean shipping
  the unsafe default for longer.

### Fold every kwarg with no deny-list at all

Rejected because:
- A genuinely transport-only kwarg (batch sizing and the like) would then
  fragment identity for reads that return identical rows — conservative, so not
  dangerous, but pointlessly so, and it would push users toward passing kwargs
  inconsistently to get cache hits. The deny-list keeps the escape hatch while
  making its use visible.

### Leave it; document the sharp edge

Rejected because:
- It is a silent wrong answer through the public cache API, in the most-used
  read path, with no error and no warning. Documentation does not convert a
  wrong answer into a diagnosable one.

## Consequences

### Positive

- The ADR-0015 rule holds for path-ful reads: a kwarg that changes returned rows
  changes the hash.
- The default direction flips. A kwarg nobody thought about is now a spurious
  miss, not a stale hit.
- One discipline across all reads: path-less
  (ADR-api-relations-are-pathless-read-ops), rest config
  (ADR-rest-config-contract-identity-folded-residence-either) and path-ful
  reads all derive identity membership by exclusion.
- Snapshot and mtime keys are fixed by the same change, so the two cannot drift.

### Negative

- **Hash movement, unavoidable and partly the point.** Reads carrying only
  `hash_path` + `table_name` emit an identical tuple and do not move — that
  covers bare `deferred_read_parquet` and therefore the build-stability
  goldens. Reads with parse kwargs move, including every duckdb
  `deferred_read_csv` (the duckdb branch injects `columns`). For the colliding
  reads the recompute *is* the fix: today two of the three in the reproduction
  are being served the wrong rows.
- **A new loud failure mode.** A newly-folded kwarg whose value has no dasher
  normalize rule now raises at tokenize, where it was previously dropped in
  silence. That is a break rather than a spurious miss, and it will surface on
  upgrade for anyone passing an exotic kwarg value. Accepted as the correct
  direction — an unhashable identity input should be an error, per ADR-0016's
  registration-tripwire discipline — but it must be in the release notes, not
  discovered.
- **`READ_TRANSPORT_KEYS` is a standing hazard with a guard.** It is the one
  place where a future edit can silently re-open under-hashing. The membership
  pin makes that edit loud; nothing makes it impossible.
- Kwarg *order* and spelling now reach identity, so two constructions of the
  same logical read that differ only in argument order tokenize apart —
  conservative, but a new source of avoidable misses if callers are
  inconsistent.

## References

- ADR-0015 (build vs cache hash; the rule this restores), ADR-0006 (hash_path /
  read_path split, whose properties this preserves), ADR-0016 (registration
  tripwires; the membership pin),
  ADR-rest-config-contract-identity-folded-residence-either
  (`identity_field_names` — derived
  membership and the argument for its direction),
  ADR-api-relations-are-pathless-read-ops
  (`normalize_read_source_identity` — the deny-list shape this generalizes)
- [#2206](https://github.com/xorq-labs/xorq/issues/2206) — the defect, with the
  reproduction
- #2070 — where `READ_IDENTITY_KEYS` was introduced, with relocatable reads
