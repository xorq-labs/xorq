# ADR-2216: REST resource reads register as lazy DataFusion tables on an owned
# connection

- **Status:** Accepted
- **Date:** 2026-07-25 (accepted 2026-08-05)
- **Deciders:** Dan Lovell

## Context

ADR-rest-config-contract-identity-folded-residence-either established
`RestBackend(PandasBackend)`: a resource read is a path-less `Read` op whose
`make_dt` boundary runs `fetch_resource`, which paginates the API into a pandas
frame and stashes it in `self.dictionary`. That ADR accepted, as an explicit
negative consequence, that "`PandasBackend` memory semantics now apply to every
config'd API; large reads want `.cache()` to parquet or param-partitioned reads
... at catalog scale."

The streaming-bare-reads change was the first time that negative bit.
`con.read(...).into_backend(other)` should stream page-wise — the consumer half
(StreamCache / RemoteTable, ADR-0013) already existed — but `make_dt` eagerly
materialized every `Read` into pandas before any batch pull could reach the
paginator. That change closed the producer half with a workaround: a
backend-optional `read_to_pyarrow_batches(expr)` seam plus an api-level fast
path in `to_pyarrow_batches` that intercepts a *bare* `Read` and calls the seam
*before* the transform pipeline runs, sidestepping the eager `make_dt` (the
`api.py` "pandas read is not lazy" FIXME).

Review surfaced three seams that workaround introduced: the streaming decision
lived in two places with divergent predicates (api-level `params or kwargs` vs
backend `params is None and limit is None`); `read_to_pyarrow_batches` called
`self._engine.fetch_batches(...)`, a method the `Engine` protocol did not
declare, so an alternative engine (dlt) would `AttributeError`; and the
api-level `chunk_size` was silently dropped on the fast path. All three existed
only because the read was eager and had to be sidestepped from above.

The root cause is the execution substrate, not the seam. `RestBackend`
inherited its `PandasBackend` base from the Phase 1/2 mixpanel backend it
generalized; ADR-rest-config-contract-identity-folded-residence-either's
decision drivers were all about the *config/identity* layer, never execution.
Reasonable people could pick pandas (simple, JSON maps naturally to frames) or
a lazy engine (streaming, bounded memory) — so the substrate choice deserved
its own decision.

## Decision drivers

- Retire ADR-rest-config-contract-identity-folded-residence-either's
`PandasBackend`-memory negative: a large resource read
  must not require holding the whole result in memory.
- One code path, not a fast path plus a fallback with divergent predicates.
- Preserve ADR-rest-config-contract-identity-folded-residence-either identity
exactly: streaming is transport, never identity;
  `Read` hashes (this backend's profile + folded per-resource config hash)
  unchanged.
- Do not inherit a Backend contract the rest backend spends effort
  suppressing (it already raises on `create_table`/`drop_table`/`create_view`).

## Decision

### Own a DataFusion connection; register resource reads lazily into it

`do_connect` constructs a private `self._df`. `fetch_resource` (the `make_dt`
boundary) no longer materializes a pandas frame. It builds a lazy
`pa.RecordBatchReader` over the engine's chunk stream (`_resource_reader`),
wraps it for replay (`_replay_cache`, below) and registers it via
`self._df.read_record_batches(cache, table_name=..., schema=...)`, returning
the resulting table.

**Which DataFusion backend matters.** Use xorq's own
(`xorq.backends.xorq_datafusion`), not the vendored-ibis
`xorq.backends.datafusion`. The vendored one registers a `RecordBatchReader`
through `_read_in_memory` → `source.read_all()`, which is fully eager and would
silently defeat this entire change. `xorq_datafusion.read_record_batches`
registers via `con.register_record_batch_reader` and casts batches lazily as
DataFusion consumes the reader.

The registration is genuinely lazy, and the harness (below) pins it: building
the read, running `make_dt`, and constructing `to_pyarrow_batches` all issue
**zero** HTTP calls in every observed case; the paginator fires only when the
engine pulls batches. `into_backend` streams page-wise.

Because the read is lazy at the engine, the producer workaround is deleted:
`read_to_pyarrow_batches`, the `RestBackend.to_pyarrow_batches` override, and
(on the core-enablers entry) the api-level `_maybe_streaming_read_reader`
interceptor plus its call site and the follow-up fix that made an explicit
`chunk_size` opt out of it. Two of the three review seams evaporate with it —
one path, one engine, no divergent predicates. The third is resolved by
promotion rather than deletion: `fetch_batches` is now *declared* on the
`Engine` protocol, because the read path depends on it, so an alternative
engine is told what it owes instead of failing with `AttributeError`.

The override-resource case folds in uniformly: `FetchOverrideEngine` implements
`fetch_batches` as a generator yielding the override's single frame, so there
is no "returns None, take the materializing path" branch, and an override read
is as lazy at construction as a paginated one.

### Multi-scan safety requires a replayable source

A bare `pa.RecordBatchReader` registered as a DataFusion table is **one-shot**.
If the physical plan scans that table more than once — a self-join, one `Read`
referenced twice, any re-scanning plan — the second scan gets an exhausted
reader and **silently returns no rows**. No error, no warning.

So the reader is wrapped in a `batchcorder.StreamCache`, which
`read_record_batches` recognizes and registers as a `CastingStreamCache`: a
replayable view over one cache that retypes on each read, so DataFusion's
repeated scans share a single buffer. `StreamCache` ingests the upstream lazily
on demand and consumes it exactly once, which is what keeps laziness intact.

Two constraints come with that path:

- **It retypes, it cannot project.** `cast` requires matching field names, so
  the reader must already emit exactly the declared schema's columns.
  `frame_from_records` reindexes each page to the declared schema and
  `pa.RecordBatch.from_pandas(..., schema=...)` enforces it at the batch, so
  this holds by construction; the harness's `schema` and `make_dt[*].schema`
  observations are what keep it honest.
- **`max_readers` is left unset.** It is the bound that lets `StreamCache`
  evict, but the scan count is not knowable at this boundary. The RemoteTable
  path can set it because it derives the count from a *compiled* SQL plan
  (`count_remote_table_readers`); `fetch_resource` is called per-`Read` with no
  view of its consumers, so an estimate would be a guess, and an under-estimate
  evicts batches a later scan still needs. Retain everything.

### Retention is paid to disk, not to RAM

Replay means retaining every batch, which partially gives back the
bounded-memory property this ADR exists to win. That trade-off is real and is
resolved deliberately rather than ignored. Measured on a 430 MB stream:

| source | peak RSS | multi-scan |
| --- | --- | --- |
| bare `RecordBatchReader` | 115 MB | **unsafe** (silently empty) |
| memory-only `StreamCache` | 561 MB | safe |
| disk-backed `StreamCache`, 32 MiB hot layer | 164 MB | safe |

Memory-only replay would make the central claim of this ADR false — RAM would
still grow with the result, which is exactly
ADR-rest-config-contract-identity-folded-residence-either's negative. So the
cache is disk-backed with a bounded hot layer (`spill_memory_capacity`, 128
MiB) and `write_policy="on_eviction"`, which means a result smaller than the
hot layer never touches disk at all — the overwhelmingly common case, including
every test — while a larger one spills and keeps RAM bounded.
`spill_disk_capacity` (64 GiB) is a diagnosable ceiling rather than an
invitation to fill the volume; exceeding it raises. Both are class attributes,
so an API whose reads are known-small or known-enormous can retune them without
touching the read path. The spill directory is per-connection, created on first
use and removed by a `weakref.finalize` when the backend is collected. The
*bound*, though, is per-read: each `fetch_resource` call builds its own cache
with its own hot layer, so a session's RAM ceiling is (number of distinct
reads) x 128 MiB rather than 128 MiB. See the retention-lifetime negative
below.

What is honestly *not* recovered: a bare reader streams in constant memory, and
a replay cache does not. Disk footprint is O(result). The trade is bounded RAM
and correct multi-scan results for disk I/O on large reads, and it is not
optional — the alternative is being silently wrong.

### The owned connection is single-partition

Every input to it is one page-wise `RecordBatchReader`, so DataFusion's
repartitioning parallelizes nothing it can exploit; what it does do is insert a
buffering shuffle in front of a stream this ADR exists to keep bounded, and
make row order and batch boundaries a function of thread scheduling — a
two-read join returned its 150 rows in a different order on every run.
Deterministic order is worth more than parallelism on an API-latency-bound
read: it is what makes a `.cache()` of a rest expression reproducible rather
than merely correct. Parallel compute over a resource read is available the
documented way, by `into_backend`-ing it onto a full connection.

Note that single-partition does not mean single-threaded: DataFusion still
polls two reader-backed scans on separate threads, so two reads on one
connection paginate concurrently through one `NativeEngine`. With an
engine-lifetime `requests.Session` — documented non-thread-safe — that put
every two-read expression on the racy side of a shared connection pool and
cookie jar. The hazard was not introduced here (the deleted workaround's
page-wise readers could already be drained concurrently) but it landed on the
default path, and it is now closed with per-read transport: `NativeEngine`
builds one session per `fetch_batches` call, so keep-alive still spans the
pages of a pagination and no two paginations share a session. A per-request
lock was the alternative and is worse: it leaves pagination state interleaved,
and a lock held across a generator's `yield` can deadlock a plan that
interleaves both scans — a hang rather than a failure.

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
and `read_identity_parts` still folds the api-wide and per-resource content
hashes (ADR-rest-config-contract-identity-folded-residence-either). `make_dt`
now returns a table whose `op().source` is the owned `self._df` rather than the
rest con — a deliberate substrate swap at the boundary — but nothing serialized
changes: the owned connection is a private implementation detail, never
captured in a profile or build artifact.

This is *checked*, not asserted. A differential observation harness
(`test_differential_substrate.py`) records `tokenize`, `get_expr_hash` and the
12-char build directory name for eleven expressions across the self-service
rest, curated github and override-only mixpanel backends, against a committed
baseline. All three are byte-identical across the substrate swap, as are
`read_source_backend_names`, `read_method_names` and the fact that computing
identity touches no network.

## Alternatives considered

### Keep the api-level interceptor, patch its seams

Add a `getattr` guard so engines without `fetch_batches` fall back, and make
the fast path honor/refuse `chunk_size`.

Rejected because it hardens a workaround for eager `make_dt` instead of
removing the eagerness. The two-places-drift, the off-protocol engine call, and
the `chunk_size` gap are all symptoms of intercepting from above.

### Register the bare reader without a replay cache

Rejected: single-scan, and it fails *silently*. A self-join or a doubly
referenced `Read` returns no rows from the second scan with no error. Constant
memory is not worth a wrong answer, and the failure mode is undetectable
without exactly the multi-scan cases the harness now pins.

### Choose bare-vs-replayable conditionally on scan count

Rejected: not knowable. `fetch_resource` runs per-`Read` at `make_dt` with no
view of the enclosing plan, and scan count needs a compiled one. Reconsider if
the deferred-reads pass ever gains access to the compiled plan.

### Memory-only replay cache

Rejected on measurement (table above): 561 MB peak for a 430 MB result. It is
replay-safe but leaves
ADR-rest-config-contract-identity-folded-residence-either's memory negative in
place, which is the negative this ADR exists to retire.

### Subclass the DataFusion `Backend` instead of composing

Rejected: it re-exposes a large read/write/SQL surface the rest backend
explicitly forbids, forcing re-suppression, and muddies rest's identity with
DataFusion's. Composition gets the same lazy execution with less surface.

### Set the DataFusion batch size to the api `chunk_size`

`datafusion.execution.batch_size` is a SessionConfig option on the owned
connection. Deferred: it is per-context, not per-call, while `chunk_size` is a
per-call arg; and `xorq_datafusion.to_pyarrow_batches` ignores its own
`chunk_size` regardless. Honoring a per-call `chunk_size` wants a rebatch
wrapper (engine-agnostic) and is out of scope for the substrate decision.

## Consequences

### Positive

- ADR-rest-config-contract-identity-folded-residence-either's
`PandasBackend`-memory negative is retired: reads stream
  page-wise and RAM is bounded by the hot layer rather than by the result.
- One execution path. The api-level interceptor, the `read_to_pyarrow_batches`
  seam and the `to_pyarrow_batches` override are gone, so the guard drift and
  the dropped `chunk_size` have nowhere to occur; the off-protocol
  `fetch_batches` is fixed by declaring it on the `Engine` protocol.
- `.limit(1)` over the three-page fixture makes 2 requests, not 4. The scan
  stops early; the "no LIMIT pushdown" claim of this ADR's proposal was wrong.
- A self-join over a resource read executes at all. It was
  `OperationNotDefinedError: No translation rule for SelfReference` on the
  pandas substrate; it now returns its rows from a single pagination.
- `GROUP BY` keeps its NULL group. `group_by("flag")` returned 2 rows because
  pandas' groupby drops NULL keys; it returns 3 now, recovering rows that were
  silently dropped. Pre-existing groups' aggregates are unchanged.
- Nullable-int materialization is self-consistent. `qty` (declared int64, with
  nulls) came back `float64`/`nan` bare but `Int64`/`<NA>` projected; it is
  `float64`/`nan` in both now.
- Override resources take the same path as paginated ones — one fewer branch.
- Rest joins the DataFusion-based streaming transport (ADR-0013/0014) the rest
  of xorq already runs on, instead of maintaining a parallel pandas island.

### Negative

- **Replay retains.** Disk footprint is O(result) on reads larger than the hot
  layer, and the constant-memory streaming of a bare single-scan reader is not
  recovered. See the trade-off section; the alternative is silent wrongness.
- **Retention is bounded per read, not per connection.** Each `fetch_resource`
  call builds a fresh `StreamCache` with its own hot layer (capped at
  `spill_memory_capacity`, 128 MiB) and registers it as a table on the owned
  connection. Nothing releases it, so N *distinct* resource reads in one
  long-lived session leave N live caches, N registered tables and N spill
  subdirectories: the RAM ceiling is N x 128 MiB, and spill (for the reads that
  exceed the hot layer at all) accumulates across reads rather than being
  O(one result). Release is tied to the connection, by
  two separate mechanisms that happen to coincide: the hot layer goes when the
  last reference to the cache does, which is when the owned `self._df` and its
  registered tables are dropped with the backend, and the spill *directory*
  goes when `_spill_root`'s `weakref.finalize` fires. Re-executing one
  expression is the already-bounded case -- the `Read`'s `table_name` is fixed
  at construction and `read_record_batches` deregisters that name before
  re-registering, which drops the previous cache -- so the unbounded axis is
  distinct reads, not repeated execution.

  Rest caches are also the one materialized resource with **no owner in the
  transform scope**: `_make_deferred_reads_replacer` calls `node.make_dt()` and
  adopts nothing, while the remote pass adopts its reader, cache and
  placeholder table into the `RemoteTableScope` that `to_pyarrow_batches` closes
  on reader exhaustion. The normal release path therefore does not reach these
  at all. The fix is to adopt the registered table and its cache into that
  scope, so both are dropped when the result reader is exhausted; it is a
  follow-up because the deferred-reads pass is `expr/api.py`, core rather than
  rest.
- **`make_dt` source swap**: the resolved table's `op().source` is the owned
  `self._df`, not the `Read`'s `source`. `make_dt` is the deferred→concrete
  boundary so this is sound, but it breaks the csv/parquet invariant that
  `dt.source is read.source`; any future code assuming that must account for
  rest.
- **Execution semantics move pandas → DataFusion** for compute over a rest
  table in place. `chunk_size` is now ignored rather than honored by opting out
  of the streaming path (xorq_datafusion's engine-wide behaviour, inherited),
  so `peak_batch_rows` is page-sized rather than chunk-sized; nullable ints
  materialize as float64; and null representation is Arrow's.
- **No request parallelism, by choice**: the owned connection is
  single-partition. Parallel compute over a resource read wants
  `into_backend`.
- **Concurrent pagination through one `requests.Session`** is now on the
  default path (see above). Pre-existing, but it should be closed.
- **`self.dictionary` is no longer the materialisation ledger**: `list_tables`
  / `table` / `get_schema` still consult it, and it now only ever holds tables
  registered on the backend directly. Those branches still work; repointing
  them at `self._df` is a follow-up.

## References

- ADR-rest-config-contract-identity-folded-residence-either (base class +
memory negative this supersedes), ADR-api-relations-are-pathless-read-ops,
ADR-build-artifacts-are-credential-free,
  ADR-0015, ADR-0014, ADR-0013
- `python/xorq/backends/rest/tests/test_differential_substrate.py` — the
  differential harness and its committed baseline; the identity, row-content
  and laziness claims above are its assertions, not prose.
