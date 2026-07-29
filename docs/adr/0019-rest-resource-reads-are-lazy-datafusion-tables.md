# ADR-0019: REST resource reads register as lazy DataFusion tables on an owned connection

- **Status:** Proposed
- **Date:** 2026-07-25
- **Deciders:** Dan Lovell

## Context

ADR-0018 established `RestBackend(PandasBackend)`: a resource read is a
path-less `Read` op whose `make_dt` boundary runs `fetch_resource`, which
paginates the API into a pandas frame and stashes it in `self.dictionary`.
That ADR accepted, as an explicit negative consequence, that "`PandasBackend`
memory semantics now apply to every config'd API; large reads want `.cache()`
to parquet or param-partitioned reads ... at catalog scale."

The streaming-bare-reads change (commit `f15361a5`) was the first time that
negative bit. `con.read(...).into_backend(other)` should stream page-wise —
the consumer half (StreamCache / RemoteTable, ADR-0013) already existed — but
`make_dt` eagerly materialized every `Read` into pandas before any batch pull
could reach the paginator. That change closed the producer half with a
workaround: a backend-optional `read_to_pyarrow_batches(expr)` seam plus an
api-level fast path in `to_pyarrow_batches` that intercepts a *bare* `Read`
and calls the seam *before* the transform pipeline runs, sidestepping the
eager `make_dt` (the `api.py` "pandas read is not lazy" FIXME).

That workaround shipped, but review surfaced three seams it introduced: the
streaming decision lived in two places with divergent predicates (api-level
`params or kwargs` vs backend `params is None and limit is None`);
`read_to_pyarrow_batches` called `self._engine.fetch_batches(...)`, a method
the `Engine` protocol does not declare, so an alternative engine (dlt) would
`AttributeError`; and the api-level `chunk_size` was silently dropped on the
fast path. All three exist only because the read is eager and had to be
sidestepped from above.

The root cause is the execution substrate, not the seam. `RestBackend`
inherited its `PandasBackend` base from the Phase 1/2 mixpanel backend it
generalized; ADR-0018's decision drivers were all about the *config/identity*
layer, never execution. Reasonable people could pick pandas (simple, JSON maps
naturally to frames) or a lazy engine (streaming, bounded memory) — so the
substrate choice deserves its own decision.

## Decision drivers

- Retire ADR-0018's `PandasBackend`-memory negative: a large resource read
  must not require holding the whole result in memory.
- One code path, not a fast path plus a fallback with divergent predicates.
- Preserve ADR-0018 identity exactly: streaming is transport, never identity;
  `Read` hashes (this backend's profile + folded per-resource config hash)
  unchanged.
- Do not inherit a Backend contract the rest backend spends effort
  suppressing (it already raises on `create_table`/`drop_table`/`create_view`).

## Decision

### Own a DataFusion connection; register resource reads lazily into it

`do_connect` constructs a private `self._df = DataFusionBackend().connect()`.
`fetch_resource` (the `make_dt` boundary) no longer materializes a pandas
frame. It builds a lazy `pa.RecordBatchReader` over the paginator's page
stream (`_resource_reader`) and registers it via
`self._df.read_record_batches(reader, table_name=..., schema=...)`, returning
the resulting DataFusion table.

DataFusion's `register_record_batch_reader` scans the reader on demand —
**verified lazy**: building the read, running `make_dt`, and constructing
`to_pyarrow_batches` all issue zero HTTP calls; the paginator fires only when
the reader is drained. So `make_dt` yields a genuinely lazy table, the
transform pipeline no longer materializes, and `into_backend` streams
page-wise, holding one page at a time.

Because the read is now lazy at the engine, the entire producer workaround is
deleted: `read_to_pyarrow_batches`, the `RestBackend.to_pyarrow_batches`
override, and the api-level `_maybe_streaming_read_reader` interceptor plus
its call site (net −88/+47 across the two files; the api-level fast path,
−29, gone entirely). The three review seams evaporate with it — one path,
one engine, `chunk_size` honored by DataFusion's own batching, no
`fetch_batches`-off-protocol assumption.

The override-resource case folds in uniformly: `_resource_reader` yields the
override engine's single frame as one batch, so there is no longer a
"returns None, take the materializing path" branch to reason about.

### Composition, not inheritance

The owned connection is held, not subclassed. `RestBackend` keeps its
`PandasBackend` base for Backend plumbing (profile machinery, `do_connect`,
`_filter_with_like`) but delegates *execution and storage* to `self._df`.
Subclassing the DataFusion `Backend` would re-expose `read_csv`,
`create_table`, SQL compilation, and UDF registration — surface the rest
backend is read-only and resource-shaped, and already suppresses. Delegation
keeps rest's identity (`name = "rest"`, its profile) while borrowing
DataFusion's execution.

### Identity is untouched

The `Read` op still has `method_name="fetch_resource"`, `source=<rest con>`,
and `read_identity_parts` still folds the per-resource `content_hash`
(ADR-0018). `make_dt` now returns a table whose `op().source` is the owned
`self._df` rather than the rest con — a deliberate substrate swap at the
boundary — but nothing serialized changes: the owned connection is a private
implementation detail, never captured in a profile or build artifact.
Verified: all 30 rest tests pass unchanged, including
`test_fetch_empty_result_keeps_declared_dtypes` (the schema-carrying reader
keeps declared dtypes on an empty result) and the two streaming tests.

## Alternatives considered

### Keep the api-level interceptor, patch its seams

Add a `getattr` guard so engines without `fetch_batches` fall back, and make
the fast path honor/refuse `chunk_size`.

Rejected because:
- It hardens a workaround for eager `make_dt` instead of removing the
  eagerness. The two-places-drift, the off-protocol engine call, and the
  `chunk_size` gap are all symptoms of intercepting from above; the lazy-read
  substrate deletes the cause. (Retained only as the minimal patch if this
  ADR is deferred.)

### Subclass the DataFusion `Backend` instead of composing

Change the base class from `PandasBackend` to the DataFusion `Backend`.

Rejected because:
- It re-exposes a large read/write/SQL surface the rest backend explicitly
  forbids, forcing re-suppression, and muddies rest's identity with
  DataFusion's. Composition gets the same lazy execution with less surface.

### Set the DataFusion batch size to the api `chunk_size`

`datafusion.execution.batch_size` is a SessionContext option, settable on the
owned connection.

Deferred because:
- It is per-context, not per-call, while `chunk_size` is a per-call arg; and
  `xorq_datafusion.to_pyarrow_batches` currently ignores its own `chunk_size`
  regardless. Honoring a per-call `chunk_size` wants a rebatch wrapper (engine
  -agnostic) and is out of scope for the substrate decision.

## Consequences

### Positive

- ADR-0018's `PandasBackend`-memory negative is retired: reads stream
  page-wise (verified lazy), bounded to one page at a time through the
  transport, not the whole result held in `self.dictionary`.
- One execution path. The api-level interceptor, the `read_to_pyarrow_batches`
  seam, and the `to_pyarrow_batches` override are all deleted; the three
  review findings against `f15361a5` (guard drift, off-protocol
  `fetch_batches`, dropped `chunk_size`) no longer have anywhere to occur.
- Rest joins the DataFusion-based streaming transport (ADR-0013/0014) the
  rest of xorq already runs on, instead of maintaining a parallel pandas
  island.
- Override resources take the same path as paginated ones — one fewer branch.

### Negative

- **`make_dt` source swap**: the resolved table's `op().source` is the owned
  `self._df`, not the `Read`'s `source`. `make_dt` is the deferred→concrete
  boundary so this is sound, but it breaks the csv/parquet invariant that
  `dt.source is read.source`; any future code assuming that must account for
  rest.
- **Execution semantics move pandas → DataFusion** for compute over a rest
  table in place (`con.read("x").filter(...).execute()`): dtype/null/string
  behavior is now DataFusion's. Current tests pass, but this is a real
  surface for expressions built directly on rest reads.
- **`self.dictionary` storage path is now vestigial**: `list_tables` /
  `table` / `get_schema` still reference it and no longer populate it; those
  branches should be repointed at `self._df` (follow-up, not done in the
  prototype).
- **No LIMIT pushdown**: the scan is lazy but DataFusion drains the reader —
  `limit(1)` over the two-page fixture still made both HTTP calls. Predicate/
  limit pushdown into the reader/table-provider is a future optimization, not
  a property this ADR delivers.

## References

- ADR-0018 (base class + memory negative this supersedes), ADR-0017, ADR-0016,
  ADR-0015, ADR-0014, ADR-0013
- Commit `f15361a5` — the streaming-bare-reads workaround this replaces
- Prototype: owned-DataFusion `fetch_resource`, 30/30 rest tests passing,
  net −88/+47 (`python/xorq/backends/rest/__init__.py`, `python/xorq/expr/api.py`)
