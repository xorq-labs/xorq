# ADR-0014: SinkNode — deferred, durable, incremental accumulation as a tee

- **Status:** Proposed
- **Date:** 2026-06-09
- **Deciders:** Daniel, Hussain
- **Context area:** `python/xorq/sinking/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`, `python/xorq/vendor/ibis/expr/types/relations.py`

## Context

xorq's caching system solves **memoization**: "have I computed this exact expression
before?" Each `execute()` of a given expression is keyed by a content hash; identical
expressions hit a cached file, different expressions miss and compute.

It does not solve **accumulation**: "append today's result to yesterday's." A daily
pipeline run produces a *different* expression every day — a new `snapshot_date`
parameter, new source data — so by construction each run is content-keyed to a
different file. There is no mechanism to append to an existing target across runs,
deduplicate re-runs, or read back accumulated history as a single expression. This is
what dbt provides with `materialized='incremental'`.

RFC 0001 (Hussain, 2026-03-22) proposes `SinkNode`, a new DAG node that — like
`CachedNode` defers a read-or-compute decision — defers a **compute-and-append**
decision to execution time, enabling incremental accumulation within xorq's
functional, immutable expression model.

The design space is genuinely contested. A `SinkNode` superficially resembles three
existing constructs that all materialise an expression into a backend table via the
same `read_record_batches`/`StreamCache` primitive (`relations.py:611`,
`storage.py:193`): `CachedNode`/`SourceStorage`, `RemoteTable`/`into_backend`, and the
proposed sink. Whether these should unify, how a sink should hash, what it evaluates
to, and how it composes with caching are all decisions reasonable people would
disagree on. This ADR records the decisions reached; the linked issues track
implementation.

## Decision drivers

- Accumulation must work across runs whose expressions differ by construction — so
  storage identity cannot be content-derived.
- The expression token must be stable across runs, or build/catalog dedup breaks.
- Must compose with the existing caching system and the planned write-audit-publish
  (WAP) pattern, without re-importing the data-loss and write-suppression footguns
  those compositions expose.
- Keep Phase 1 minimal and honest about its gaps rather than implying guarantees
  (concurrency, schema evolution) it does not provide.

## Decision

### SinkNode and CachedNode are duals, not subtypes

A `CachedNode`'s identity is a function of its parent's **content**: `f(content)`. Same
expression → same key → cache hit. It answers *"have I computed this exact thing?"*

A `SinkNode`'s identity is a **stable, user-supplied name** (the target path / table
name), independent of the parent's content hash. It answers *"where do I accumulate,
regardless of what flows in?"* This is irreducible: if a sink were content-keyed, every
run — being a different expression — would write to a different target and nothing would
ever accumulate.

So the two are **duals on identity**: a cache *derives* the name from content; a sink
*fixes* the name and lets content vary. They share resolution *machinery* (a deferred
DAG node resolved at execution time via `op.replace()`, and a strategy/storage split),
but neither is a special case of the other. The ADR explicitly rejects the framing that
"a CachedNode is a SinkNode whose name is the key."

| | identity (name) | lifetime | question answered |
| -- | -- | -- | -- |
| `CachedNode` / `SourceStorage` | content hash | durable until `drop` | "computed this exact thing before?" |
| `RemoteTable` / `into_backend` | `gen_name()`, anonymous | ephemeral — dropped in `clean_up` | (none — query-execution mechanic) |
| `SinkNode` / sink storage | **user-supplied stable name** | **durable, never auto-dropped** | "where do I accumulate, whatever flows in?" |

### A SinkNode is a tee, not a terminal sink

In dataflow systems a "sink" is terminal — data is consumed, nothing flows past. xorq's
`SinkNode` is deliberately **not** terminal. `.sink()` resolves to **this run's
actually-written delta** (the rows that landed after incremental-filtering and
deduplication), and passes that delta downstream. The full accumulated target is read
separately via **`sink.read_all()`**.

The write is an **unconditional side effect at transform time** — driven by
`_register_and_transform_sink_nodes`, mirroring `_register_and_transform_cache_tables`
(`api.py:212`). Whatever downstream operations do with the returned delta, the write
fires whenever the `SinkNode` is present in the executed subgraph. No downstream
operation can elide it.

This tee semantics is what makes the sink composable. The original RFC was internally
inconsistent here: its acceptance criteria said "`.sink()` returns the full accumulated
dataset" (read-all), while its dbt mapping mapped `{{ this }}` → a separate read-back
call. The two contradict. We resolve in favour of the tee:

| dbt concept | xorq equivalent |
| -- | -- |
| `materialized='incremental'` | `.sink()` |
| `unique_key` | `Sink(unique_key=(...))` |
| `incremental_strategy` | `Sink(strategy="append"\|"merge")` |
| `{{ this }}` / `SELECT * FROM {{ this }}` | `sink.read_all()` |
| `is_incremental()` | implicit — write checks whether `read_all()` is non-empty |
| `WHERE col > max(col) FROM {{ this }}` | `Sink(incremental_column="col")` |

`read_all()` (not `view()`) is the read-back name: it matches the `SinkStorage.read_all()`
method and avoids collision with the existing ibis `Table.view()` (`relations.py:887`),
which means an aliased self-reference for self-joins — an unrelated concept.

There is no atomic SQL primitive for "write then return the whole target"; read-all is
inherently a second query against the target. The tee, by contrast, maps to native
`INSERT … RETURNING *` / `MERGE … RETURNING` — one statement, atomic. This is a further
reason to make the tee the default and read-all an explicit separate call.

### Hashing: config only, never accumulated state

The `SinkNode` token (`normalize_sink_node`, parallel to `normalize_cached_node`) folds
in **`(parent_token, sink_config)` only**, where `sink_config` is the storage type,
target name/path, `unique_key`, `strategy`, and `incremental_column`. It must **never**
fold in accumulated data, `read_all()` contents, row counts, or
`max(incremental_column)` — any runtime state would change the token every run and
break "same pipeline" detection for build artifacts and catalog dedup.

`unique_key`, `strategy`, and `incremental_column` **are** part of the token: changing
dedup or accumulation semantics is a real change to pipeline identity.

Consequently, *"has this exact result been computed?"* is the wrong frame for a sink. A
sink is an accumulator; "computed" is not a yes/no. The token answers *"is this the same
pipeline definition?"*; **idempotency** ("does this run's input already exist?") is a
write-time property guaranteed by the `unique_key` anti-join, not by the hash.

### Incremental loading

`Sink.write(value)`:

- **Incremental filter** — if `incremental_column` is set, filter `value` to rows newer
  than the current maximum of that column in `read_all()`. On the first run `read_all()`
  is empty, so everything is written (the implicit `is_incremental()`).
- **append** — anti-join on `unique_key` to skip rows already present, then
  `storage.append()`. New keys inserted, existing skipped.
- **merge** — anti-join existing rows out of `read_all()`, union with new, then
  `storage.replace()`. New keys inserted, existing overwritten.

**Boundary rule.** The incremental filter is **`>= max` when `unique_key` is set** and
**`> max` when it is not**. Strict `>` everywhere has a silent data-loss footgun: a
late-arriving row whose `incremental_column` equals the current maximum is never
`> max`, so it is dropped forever once the watermark advances past it. Using `>= max`
re-pulls the boundary rows and the `unique_key` anti-join dedups the overlap →
exactly-once, no loss. Without a `unique_key` there is nothing to dedup on, so `>= max`
would duplicate the boundary; we fall back to `> max` (best-effort, documented). Users
are advised to always pair `incremental_column` with `unique_key`.

**Tee return.** `.sink()` returns the rows **actually written** (post-filter,
post-dedup). An idempotent re-run writes nothing and returns an empty result — the
no-op is observable.

### Composition with caching: upstream allowed, downstream forbidden

- **Cache upstream** (`expr.cache(c).sink(s)`) — **allowed and canonical.** Bottom-up
  resolution memoises the expensive compute first, then the sink writes that result's
  delta. The sink itself is never cached, so its write always fires.
- **Cache downstream of a sink** (`expr.sink(s).cache(c)`, or any `CachedNode` that is an
  ancestor of a `SinkNode`) — **forbidden; raises at construction time.** Two hazards,
  both rooted in the sink token being stable across runs (above):
  1. *Write-suppression* — the downstream cache key is stable, so from run 2 onward the
     cache hits and `set_default` returns the stored value without descending into its
     parent; the `SinkNode` is pruned before the sink pass sees it and the write never
     fires — accumulation silently freezes.
  2. *Stale tee* — even with the write forced, the cache memoises run 1's delta under a
     stable key; later runs accumulate correctly in storage but return run 1's delta
     downstream.

This enforces (rather than merely documents) the RFC's open question on composition
ordering. Caching expensive *downstream analysis* of a sink is still possible — root
that analysis on `sink.read_all()` as its own separate, cacheable expression.

### Durability and relationship to `into_backend`

Sink tables are **durable by default, never temporary, never auto-dropped**; teardown is
only the explicit `SinkStorage.drop_all()`. A `SinkNode` does **not** enter the
`created`/`clean_up` drop path that `RemoteTable` uses (`api.py:482`). This durability is
the one concrete difference from `into_backend`: `into_backend`/`RemoteTable` is an
anonymous, ephemeral query-execution mechanic; a sink is a named, durable persistence
mechanic. They share the underlying write transport but are not the same construct and
are not conflated in the API.

### Cross-source caching is orthogonal

A `SinkNode` does **not** solve or prevent the cross-source caching issue
(`maybe_prevent_cross_source_caching`, `caching/__init__.py:226`). A sink targeting a
backend different from its parent's must still relocate data across backends, via the
same `read_record_batches` path `SourceStorage.put` already uses. What changes is that a
sink's target backend is **explicitly declared** by the user, so there is no *implicit*
`into_backend` injection and no cache-key mutation — `maybe_prevent_cross_source_caching`
is **not applicable to sinks**.

### Concurrency (Phase 1: single-writer)

Phase 1 assumes a **single writer** per sink. Each `ParquetAppendStorage.append()`
writes a UUID-named temp file and **atomically renames** it, so no concurrent writer can
produce a partial or corrupt file. Re-run idempotency (sequential re-runs) is provided by
the write-time `unique_key` anti-join.

The `read view() → anti-join → append` sequence is a TOCTOU race for *concurrent*
writers: two writers may both observe the same existing keys and both append overlapping
rows. Atomic rename does not prevent this *logical* duplication. Concurrent writers to
the same sink are therefore **explicitly unsupported in Phase 1**, with documented
behaviour: no corruption, but possible duplicates (last-writer-wins). The real fixes —
engine-level `MERGE`/transactions for `SourceAppendStorage`, an optional directory lock
for `ParquetAppendStorage`, or read-time dedup-on-`unique_key` — are deferred.

### Schema: fail-fast at write and read (Phase 1)

A sink's schema is **fixed at first write**. Every subsequent write asserts the incoming
schema equals the established schema and **raises on mismatch** — silently appending
mismatched batches would corrupt `read_all()`, so the check must apply at write time, not
only at read. `read_all()` assumes a uniform schema. Schema evolution (merge-on-read, or
pin-and-cast as in `read_record_batches`' `_select_and_cast`) is deferred.

### Compaction: none in Phase 1

`read_all()` is a `deferred_read_parquet` over the whole target directory. `append`
accumulates one small file per run, degrading read performance over time; this is an
**append-only concern** — the `merge` strategy does a full `replace()` each run and so
self-compacts. An explicit `compact()` (optionally threshold-triggered) is deferred to
Phase 4.

### Naming

The node is `SinkNode`. The ADR records that "sink" carries a terminal connotation the
node deliberately violates (it is a tee), and documentation must state this up front.
`AccumulateNode` was the runner-up; `SinkNode` was kept for its established RFC usage,
standard "durable write destination" vocabulary, and clean storage-subtype names
(`ParquetSink`, `SourceSink`, `IcebergSink`).

## Alternatives considered

### Unify SinkNode and CachedNode under one node type

Treat caching as a special sink whose name happens to be the content key, with a single
node serving both.

**Rejected** because the identities are duals, not a generalisation: a cache *derives* its
name from content while a sink *fixes* its name and varies content. No single key function
serves both without breaking one of the two use cases. The shared *machinery* (deferred
node, `op.replace`, strategy/storage split) does not justify a shared *type*.

### Read-all semantics for `.sink()`

`.sink()` resolves to the full accumulated target rather than the run's delta.

**Rejected** because it breaks composition. Under WAP (`write → audit → publish`), a
publish step downstream of a read-all sink would re-write the *entire* staging history to
production every run, double-accumulating. Read-all also has no atomic SQL primitive
(it is always a second query), and contradicts the dbt `{{ this }}` mapping. Read-all is
preserved as the explicit `sink.read_all()` call.

### Permit downstream caching of a sink (document, don't enforce)

The RFC's open question left composition ordering as "enforce or document."

**Rejected** in favour of enforcement: the stale-tee hazard is semantic and unfixable, and
write-suppression silently halts accumulation. A construction-time error is safer than a
documented footgun.

### Mandate ADBC as the database write transport

**Deferred.** The write transport (ADBC vs `read_record_batches` vs native `create_table`)
is a per-`SinkStorage` implementation choice evaluated in Phase 2, not a contract of the
node semantics. Likewise the shared `write_to_source` primitive that would unify the three
existing write paths is **out of scope** for this ADR. batchcorder's role (buffering
multi-consumer reads during a write) is an implementation detail, not a normative
decision.

### Directory locking for concurrent `ParquetAppendStorage` writers in Phase 1

**Deferred.** Keeps Phase 1 minimal; concurrency support is a later phase with several
candidate mechanisms (engine `MERGE`, directory lock, read-time dedup).

## Consequences

### Positive

- Accumulation across runs becomes a first-class, immutable-expression-native operation,
  closing the gap between xorq's memoization and dbt-style incremental materialization.
- Tee semantics make sinks composable — chains and the planned WAP pattern fall out
  naturally, and `.sink()` maps to native `INSERT … RETURNING`.
- Stable, config-only hashing keeps build artifacts and catalog dedup intact across the
  repeated runs that are the whole point of a sink.
- Construction-time rejection of downstream caching turns two silent, hard-to-debug
  failure modes into an immediate, explainable error.
- Phase boundaries are honest: the gaps (concurrency, schema evolution, compaction) are
  documented with their resolution paths rather than implied to be solved.

### Negative

- `.sink()` returning the delta while `sink.read_all()` returns the whole is a genuine
  surprise; the "sink is a tee, not terminal" point must be made loudly in docs or users
  will expect read-all from `.sink()`.
- Single-writer Phase 1 means concurrent pipelines writing the same sink can produce
  logical duplicates — acceptable only because it is documented and bounded by atomic
  rename (no corruption).
- `append` sinks accumulate small files until Phase 4 compaction lands; large
  long-running append sinks will see read performance degrade.
- Fail-fast schema handling means any change to the parent expression's schema breaks
  appends to an existing sink until schema evolution is implemented.

## References

- RFC 0001: Deferred Sinks (Hussain, 2026-03-22)
- [XOR-252](https://linear.app/xorq-labs/issue/XOR-252) — Deferred Sinks (epic)
- [XOR-262](https://linear.app/xorq-labs/issue/XOR-262) — Phase 1: core primitives
- [XOR-263](https://linear.app/xorq-labs/issue/XOR-263) — Phase 2: SourceAppendStorage, IcebergSinkStorage
- [XOR-264](https://linear.app/xorq-labs/issue/XOR-264) — Phase 3: catalog serialization
- ADR-0013: batchcorder StreamCache for RemoteTable fan-out — `docs/adr/0013-batchcorder-stream-cache-for-remote-table-fan-out.md`
- dbt incremental materialization docs
- Write-Audit-Publish pattern: https://lakefs.io/blog/data-engineering-patterns-write-audit-publish/
- `CachedNode` / `RemoteTable`: `python/xorq/expr/relations.py`
- Cache resolution: `_register_and_transform_cache_tables`, `python/xorq/expr/api.py:212`
- Cross-source caching: `maybe_prevent_cross_source_caching`, `python/xorq/caching/__init__.py:226`
