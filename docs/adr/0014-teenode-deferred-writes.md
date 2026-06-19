# ADR-0014: TeeNode, deferred write as a side effect

- **Status:** Accepted
- **Date:** 2026-06-10
- **Deciders:** Daniel Mesejo, Dan Lovell
- **Context area:** `python/xorq/writes/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`, `python/xorq/vendor/ibis/expr/types/relations.py`

## Context

Xorq has first-class **deferred reads**: `deferred_read_parquet`, `deferred_read_csv`, and
the `Read` op (`relations.py`) defer a read to execution time, invoking
`getattr(source, method_name)(...)` only when the expression runs. There is no symmetric
**deferred write**. An expression cannot, as part of running, write its rows to a target as
a side effect and keep going.

This ADR adds that: a write performed as a side effect of execution. It is implemented as a
cache-hash-neutral pass-through `TeeNode` (the cache hash ignores it; the build hash
reflects the write-through) that drives a `WriteThrough` generator. xorq commits to a single invariant —
**every node streams batches** — so there is no terminal sink node; "terminal" behavior
(data in, nothing out) is reached by composition (see Alternatives).

The write is performed through a **transport**. Phase 1 ships one: a client-side streaming
generator that round-trips the parent through Arrow. The design treats this as the *default,
portable* transport rather than the definition of a deferred write, and explicitly admits an
in-engine transport (CTAS / writable CTE) for the case where the parent and the write target
share a backend (see Alternatives). Keeping the write *what* (a hash-neutral, cache-respecting
side effect) separate from the write *how* (transport) is what lets the same construct back a
Postgres-native write-through cache later without reopening this decision.

## Decision drivers

- A write should be a **side effect** that does not change what an expression evaluates to,
  so it composes with the rest of the graph and with caching.
- The write must **respect the cache**, independent of transport. Cache *hits* are handled
  **structurally and ahead of execution**: the cache pass runs before the tee pass and prunes the
  `TeeNode` (or its in-engine target) from the graph before any write is registered or lowered, so
  no write fires on a hit regardless of transport or which side drives the pull. Orthogonally, a
  write that *is* registered fires only when its output is actually consumed — the single-puller
  property for the client-side generator (no pull → the generator never runs), or the writing
  statement being a sub-statement of the consuming query for an in-engine transport.
- A `WriteThrough` in xorq is **always a streaming write-through**: it writes each batch as a side
  effect and yields it onward. There is no terminal sink node (despite the general
  convention that "sink" means terminal); terminal behavior is reached by composition, not a
  dedicated node (see Alternatives).

## Decision

### One node: a pass-through `TeeNode`

`TeeNode` is a **pass-through**: it evaluates to its parent's rows unchanged and, as those
rows are pulled through it, **hands each batch to a write-through** as a side effect. The TeeNode
does not write anything itself; it delegates to a `WriteThrough` whose `write_through(batches)`
generator wraps the parent's batch stream. The user-facing `.tee()` builds a `TeeNode`.
Per the streaming invariant above, `TeeNode` is the only node this ADR ships.

`TeeNode` does not reference any concrete writer type. It holds a `WriteThrough` in its `writer`
field (see the `WriteThrough` contract below) and is otherwise oblivious to what the
write-through does with the batches it receives.

### `TeeNode` is hash-neutral, like `Tag`

`TeeNode` is modeled on `Tag` (`relations.py`). Its schema equals its parent's schema,
and it is **stripped before hashing** by a resolution pass that replaces it with its parent,
exactly as `_remove_non_hashing_tag_nodes` (`api.py`) does for `Tag`. So `expr.tee(s)`
and `expr` produce the **same** content hash, and a `CachedNode` above or below the tee keys
as if the tee were not there.

### The `WriteThrough` contract

A `WriteThrough` is an abstract base class with a single method:

```python
class WriteThrough(abc.ABC):
    @abc.abstractmethod
    def write_through(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Pull from batches, write each as a side effect, yield onward."""
        ...
```

`write_through(batches)` is a **generator**: it pulls from `batches`, writes each batch as a side
effect, and yields it downstream unchanged. Every `WriteThrough` is a streaming write-through by
definition — there is no terminal variant — so the `-> Iterator` signature is the whole
contract, not a special case. Each write-through subclass owns its full lifecycle
(open, commit/publish, abort/cleanup) inside its `write_through` implementation. The downstream
consumer drives iteration — the write-through never independently pulls.

This generator is the **client-side transport**, not the definition of a write-through. It is the
only transport Phase 1 ships, but when the parent and the write target live on the same backend
the write MAY instead be lowered into the engine (CTAS or a writable CTE), in which case no Arrow
batches cross the client and the `-> Iterator` interface is never invoked. The contract every
write-through owes is the **side effect** — hash-neutral, cache-respecting, write-then-yield — not
the generator shape. The cache-respecting guarantee survives the swap: for the generator it comes
from single-puller semantics; for an in-engine transport it comes from the writing statement
running exactly when the consuming query runs (see Alternatives).

This gives the cache-respecting behavior for free:

- **Single puller.** The only consumer of the stream is downstream. The write-through never
  pulls independently; it receives batches the downstream pull already produced. If downstream
  does not pull — an unconsumed or dead branch — the generator never runs and nothing is written.
  (Cache *hits* are handled earlier and structurally: the cache pass prunes the `TeeNode` before
  the tee pass registers any write — see below — so cache suppression does **not** rely on this
  runtime property.)
- **Write-then-yield.** For write-throughs that commit incrementally (`ParquetWriteThrough`, and
  `BackendWriteThrough` against a backend whose `read_record_batches` accepts a write `mode`),
  every batch handed downstream was written by the write-through first, so the written set equals
  the delivered set with no off-by-one. Bulk backends are the exception: a backend that
  cannot append must ingest in a single call, so `BackendWriteThrough._write_bulk` yields every
  batch downstream first and ingests once after the stream is exhausted. There the write
  trails delivery — a post-stream ingest failure means downstream already saw data the
  write-through never persisted. Such failures surface as drain/close errors rather than being
  swallowed.
- **Preferred: threaded.** `ThreadedBackendWriteThrough` is the preferred backend consumer
  and is what `.tee(con, table_name=...)` constructs by default: it runs the ingest on a
  background thread with an unbounded queue, decoupling the write from the downstream consumer
  so a slow write does not block the producer. The cost is buffering lag in memory (the
  unbounded queue gives up backpressure). The plain inline `BackendWriteThrough` remains
  available for callers who want the write in the pull path with bounded memory; passing it
  explicitly opts back into lock-step.

### `ParquetWriteThrough`: single-file parquet consumer

`ParquetWriteThrough(path, mode)` writes to a single, user-named parquet file. `path` is the final
file path, known at construction time. `mode` defaults to `create` (matching `BackendWriteThrough`),
so an unqualified write fails rather than silently appending to an existing target. `write_through`
stages batches to a randomized temp sibling (`tempfile.mkstemp` with a `.tmp` suffix in `path`'s
directory), then publishes atomically after clean exhaustion:

- **`create`**: publishes via `os.link(tmp, path)` — atomic create-or-fail. If `path`
  already exists the link raises `FileExistsError`; no glob, no lock needed. The temp is
  always cleaned up.
- **`append`**: acquires an exclusive lock (`flock_exclusive`, the `fcntl.flock` wrapper in
  `xorq.common.compat`) on `path + ".lock"`. If `path` already exists, it streams the existing
  file's row groups followed by the staged batches through a `ParquetWriter` into a second
  temp (`path + ".merge.tmp"`), and renames that over the target; both reads use
  `ParquetFile.iter_batches()`, so memory stays at O(batch_size) regardless of existing file
  size. If `path` does not exist, the staged temp is renamed into place directly — no merge.
  The lock serializes concurrent appenders; the rename is atomic. The lock file is
  removed after the operation.

An error or early stop mid-stream discards the temp file and publishes nothing. Object
stores have no atomic link/rename and need a different publish; that is out of scope for
Phase 1.

### `BackendWriteThrough`: per-backend ingest consumer

`BackendWriteThrough(con, table_name, mode, kwargs)` delegates writes to any xorq backend's
`read_record_batches` method. It has two execution paths, selected by whether the backend
accepts a `mode` parameter:

- **Per-batch** (backends with `mode`, e.g. Postgres via ADBC): the first batch is ingested
  with mode `create` (or `append` if mode is `append`), and all subsequent batches use
  `append`. Each batch commits immediately, so a mid-stream failure leaves earlier batches
  written — this path is **not** all-or-nothing.
- **Bulk** (DataFusion, DuckDB, Pandas — no `mode` parameter): all batches are buffered in
  memory and registered as a single table after the stream is fully consumed. A mid-stream
  error means nothing is written.

Extra `kwargs` are forwarded to `read_record_batches`, allowing backend-specific options.
`kwargs` is **identity-neutral**: it is excluded from equality, hashing, and
`__dasher_tokenize__` (`hash=False, eq=False`), so two tees differing only in `kwargs`
hash identically and key the same cache entry and build artifact. This is sound only because
`kwargs` is meant to tune write *mechanics* (compression, batch size, ADBC options), not the
logical result. Anything that changes the rows written must not be passed through `kwargs`,
or it will silently escape the hash (see Consequences).

`ThreadedBackendWriteThrough` is a `BackendWriteThrough` subclass that replaces both the per-batch
round-trips and the bulk buffer with a single streaming `read_record_batches` call on a
background thread. A `queue.SimpleQueue` bridges the generator and the thread: each batch
is pushed onto the queue and yielded downstream; the thread's reader drains the queue. The
queue is unbounded (no backpressure) — a bounded queue would deadlock if the write-through thread
died without draining. Rollback on error follows the backend's own semantics.

### `WritePrimaryWriteThrough`: the write-primary transport

`ThreadedBackendWriteThrough` is **downstream-primary**: the downstream pull drives the write,
which can lag behind (unbounded queue) but never lead, and an early-stopping downstream needs
`drain=True` to finish the write. `WritePrimaryWriteThrough(inner, maxsize=0)` is the mirror — a
generic wrapper around any `WriteThrough` where the **write** owns the pull loop on a background
thread and downstream consumes the already-written batches through a single 1:1 queue (no fan-out;
the write and the pass-through are the same batch sequence at the same position). Two consequences
follow:

- **Drain-always.** Because the write thread drives the pull, it runs to completion independently
  of the downstream consumption rate. An early stop cannot abort it: on `GeneratorExit` the
  generator keeps draining (and discarding) the queue so the write thread finishes. This is the
  same drain-always shape the future in-engine transport has (see Alternatives) — `WritePrimaryWriteThrough`
  is a shipped instance of it, and like that transport it cannot honor `drain=False`.
- **Backpressure dial.** `maxsize` tunes the bridge queue: `0` (default) is unbounded (the write
  races ahead, downstream may lag in memory); `>0` is a bounded queue where a full queue blocks
  the write — i.e. downstream back-pressures the write. This is the one shipped consumer that can
  bound memory while still decoupling the write onto its own thread.

`maxsize` is **identity-neutral** (`hash=False, eq=False`) and `__dasher_tokenize__` delegates to
the inner writer, so wrapping a writer changes neither the cache key nor the build artifact — the
same principle as `drain`.

### `.tee()`: the user-facing method

`.tee()` is polymorphic — it accepts either a `WriteThrough` directly or a backend connection:

```python
def tee(self, target: WriteThrough | BaseBackend, *, table_name=None, drain=True, **kwargs) -> Table
```

When `target` is a `BaseBackend` (with `read_record_batches`), `.tee()` auto-creates the
preferred `ThreadedBackendWriteThrough` from `table_name` and the remaining `**kwargs`. When
`target` is already a `WriteThrough`, extra keyword arguments are rejected.

### `drain` (default `True`): completing the write on early stop

By default (`drain=True`), the execution pipeline wraps the write-through generator in a
`DrainingIterator` (`python/xorq/writes/write_through.py`) so a downstream early-stop
(`LIMIT`/`head`) still consumes the remaining batches through the writer on a background
thread and the write completes. With `drain=False`, an early-stop instead aborts the
write-through — the generator sees `GeneratorExit`, cleanup runs, and nothing is published.

During normal iteration, `DrainingIterator` is a transparent pass-through. After downstream
execution completes, the pipeline calls `close()` on each `DrainingIterator`. If the
generator was not fully consumed, `close()` spawns a non-daemon background thread that
continues iterating the generator — each `next()` triggers the write-through's write side-effect as
usual, but the yielded batches are discarded since no downstream consumer remains. The
pipeline then calls `join()` to wait for the drain thread before dropping temporary tables
(so the parent reader stays valid during the drain).

The key invariant that makes this safe: the "no independent pulling" constraint exists to
avoid buffering for downstream. Once downstream is done, there is no downstream to buffer
for, so the write-through can freely consume the remainder of the parent stream.

If the drain thread encounters an error, `join()` re-raises it to the caller. The lifecycle
is explicit — there is no `__del__` finalizer, since the backend holds the reader alive
past GC, making finalizer-based cleanup unreliable.

No changes to the `WriteThrough` ABC or any existing write-through implementation are required.

### Fan-out of two, chained for N

Phase 1 supports a fan-out of two: one main consumer plus one inline write. A fan-out of N
is achieved by **chaining N-1 `TeeNode`s**, not by a dedicated N-way node.

### Durability and relationship to `into_backend`

A deferred write yields a **persistent, user-named, user-owned** target. It is never temporary
and never auto-dropped; teardown is explicit. This is the concrete difference from
`RemoteTable`/`into_backend`, which produces an **anonymous, ephemeral** table dropped in
`clean_up` (`api.py`). They share the underlying write transport, and a `CachedNode` is a third
user of it. What separates these constructs is **downstream wiring** and **target ownership**, not
the transport: a cache substitutes the source with a read of the target and is keyed by hit/miss;
a tee leaves the target as a side branch and passes the original rows through; `into_backend`
yields an ephemeral table. They may share execution machinery while the API keeps the surfaces
distinct — which is what leaves room for a write-through cache built on the tee transport without
collapsing the user-facing distinction.

### Node shape

```python
# python/xorq/expr/relations.py

class TeeNode(ops.Relation):
    schema: Schema            # == parent.schema; pass-through
    parent: ops.Relation
    writer: WriteThrough      # a WriteThrough whose write_through() drives the write
    drain: bool = True        # drain remaining batches on early stop (default)
    values = FrozenDict()     # hash-neutral, like Tag
```

`TeeNode` declares its identity via `__dasher_tokenize__`, which returns
`("tee-node", self.schema, self.writer)` — delegating write-through identity to each
`WriteThrough` subclass's own `__dasher_tokenize__`.  `drain` is excluded from the
token because it is purely an execution-time concern that does not change the
logical result.  The cache hash path strips `TeeNode` (like `Tag`), so
`__dasher_tokenize__` is only reached when `_hash_expr_components` explicitly
tokenizes the extracted nodes.  The build hash path (`get_expr_hash`) enters the
`include_tee_nodes()` context manager so that different write-throughs produce different
build artifacts.

The blocks below are illustrative. In the implementation `WriteThrough` is an `abc.ABC` and the
concretes are `@frozen` attrs classes (immutable, value-equal, with converters/validators — e.g.
`mode` coerces from a string, `con` is validated to expose `read_record_batches`). Immutability and
value-equality are what make a write-through safe to embed in the hashed `TeeNode`; a mutable
plain-class reimplementation would break build-hash determinism.

```python
# python/xorq/writes/write_through.py

class ParquetWriteThrough(WriteThrough):
    path: Path                # the final file path (not a directory)
    mode: WriteMode           # "create" (link) | "append" (merge + rename)
    def __dasher_tokenize__(self): return ("ParquetWriteThrough", str(self.path), self.mode)
    def write_through(self, batches): ...  # stage to path.tmp, publish on exhaustion

class BackendWriteThrough(WriteThrough):
    con: Any
    table_name: str
    mode: WriteMode
    kwargs: dict = field(hash=False, eq=False)   # identity-neutral: tunes mechanics, not rows
    def __dasher_tokenize__(self): return ("BackendWriteThrough", getattr(self.con, "name", ""), self.table_name, self.mode)
    def write_through(self, batches): ...  # per-batch or bulk, depending on backend
```

```python
# Table.tee (vendored relations.py): the user-facing attach

def tee(self, target: WriteThrough | BaseBackend, *, table_name=None, drain=True, **kwargs) -> Table:
    ...
    op = TeeNode(schema=self.schema(), parent=self.op(), writer=writer, drain=drain)
    return op.to_expr()
```

Two resolution passes keep hash neutrality and the side effect separate:

- **Hashing**: a strip pass replaces each `TeeNode` with its parent before the hash is
  computed, like `_remove_non_hashing_tag_nodes` for `Tag`. `_remove_tee_nodes`
  does the same off the SQL path.
- **Execution**: the **default transport** is `register_and_transform_tee_nodes`, which replaces
  each surviving `TeeNode` with a backend table fed by `writer.write_through(reader)`:

  ```python
  reader = parent_expr.to_pyarrow_batches()
  wrapped = pa.RecordBatchReader.from_batches(reader.schema, node.writer.write_through(reader))
  table = con.read_record_batches(wrapped, table_name=gen_name())
  ```

  This mirrors the RemoteTable pass (`register_and_transform_remote_tables`); it is portable
  across backends but always round-trips the parent through Arrow in the client. It runs after
  cache resolution, so a downstream cache hit prunes the tee before this pass sees it and the
  write never fires. When the parent and the write target share a backend, an in-engine transport
  MAY satisfy the write directly (CTAS for materialize/create, a writable CTE for
  append/pass-through), bypassing the round-trip — the same-backend short-circuit
  `SourceStorage.put` already uses for caching (`is_single_backend` → `create_table`, in
  `python/xorq/caching/storage.py`). That transport is deferred (see Alternatives); only the Arrow
  transport ships in Phase 1.

### Naming

The user-facing method is `.tee()`, matching the Unix `tee` command: data flows through and
a copy is written as a side effect. The consumer abstraction is named `WriteThrough`, and the
field on `TeeNode` is `writer`. The split is **node vs. consumer**, not pass-through vs.
terminal: `tee`/`TeeNode` is the thing in the graph, and `WriteThrough` is the streaming
write-consumer it drives. Because of the streaming invariant there is no second, terminal kind
of `WriteThrough`, so the name keeps a single, total meaning.

## Alternatives considered

### A terminal `WriteThrough` node (deferred write via `methodcaller`)

A dedicated terminal node, the true counterpart to `deferred_read_*`: data in, nothing past.
Executing it would fire `operator.methodcaller(sink_method)(parent)` at execution time,
mirroring how `Read.make_dt` resolves a deferred read via `getattr(source, method_name)(...)`.

**Rejected.** It violates the streaming invariant (every node streams batches). Terminal
behavior is reached by composition instead (see below); keeping a single streaming `WriteThrough`
concept removes the recurring "which `WriteThrough`?" ambiguity.

### Terminal write as a draining tee plus discard (`.write()`)

Rather than a terminal node, a terminal write is **composed**: write-through the tee, drain
the remainder, and discard the output. Mechanically this is `expr.tee(writer, drain=True)` with
the write-through's reader drained to exhaustion and the rows discarded — equivalent in spirit to
`expr.tee(writer, drain=True).limit(0)`, but `limit(0)` is an implementation detail, not the
surface.

**Adopted as the terminal story, exposed via sugar.** A future `.write(writer)` helper expands
to this composite. Two caveats it must document: (1) the write fires *only* because
`drain=True` forces consumption after downstream stops — a plain `tee(writer).limit(0)` is
pull-suppressed and silently writes nothing, so `limit(0)` must never be the user-facing
surface; (2) drain runs the write on a background thread joined before teardown, so a write
failure surfaces post-execution at `join()` rather than as the expression's own failure.

### In-engine (same-backend) write transport

When the parent expression and the write target live on the **same backend**, the write need not
round-trip through Arrow. Postgres (and similar SQL engines) can satisfy it in a single statement:

- **Materialize / `create`**: `CREATE TABLE target AS <parent SQL>` — terminal; downstream then
  reads the target. This is the *cache* shape, and `SourceStorage.put` already does it
  (`is_single_backend` → `create_table`, `python/xorq/caching/storage.py`).
- **Append / pass-through**: a data-modifying CTE,
  `WITH w AS (INSERT INTO target SELECT … RETURNING *) SELECT * FROM w` — writes as a side effect
  and yields the rows onward, the SQL-native analogue of `write_through`.

**Deferred — door explicitly left open.** This is not Rejected; it is the path to Postgres-native
deferred writes and write-through caching. Three things make it future work rather than Phase 1:

1. **A compile/lowering seam is required.** Today `TeeNode` is stripped off the SQL path
   (`_remove_tee_nodes`) and rewritten to an Arrow-fed table *before* SQL compilation, so it never
   reaches the compiler. An in-engine transport needs `TeeNode` (or its target) to lower to SQL
   instead of being rewritten — a new capability, not a tweak to the existing pass.
2. **Drain semantics become intrinsic, not opt-in.** Postgres data-modifying CTEs always run to
   completion regardless of an outer `LIMIT`, so this transport is **drain-always**. That matches
   the default (`drain=True`) but cannot honor an explicit `drain=False` early-stop abort.
3. **Scope is same-instance tables only.** Parquet targets (`ParquetWriteThrough`) and
   cross-backend writes still require the Arrow transport; the in-engine path is an optimization
   for the same-backend case, not a replacement.

### A single node that both writes and passes through

One node that is simultaneously a terminal sink and a pass-through tee.

**Rejected.** It conflates two behaviors with opposite modes. With every node streaming and
terminal behavior composed from a draining tee, a single pass-through `TeeNode` driving a
streaming `WriteThrough` covers both needs without a dual-mode node.

### Write unconditionally at execution, regardless of pull

Fire the write whenever a write node is present in the executed subgraph.

**Rejected.** It would fire on any branch whose output is never consumed, doing I/O nobody asked
for. Note this is **not** what gives cache-hit suppression — that is structural: the cache pass
runs before the tee pass and prunes the `TeeNode` on a hit
(`_register_and_transform_cache_tables` → `register_and_transform_tee_nodes`), so the write is
never registered regardless of transport or which side drives the pull. What tying the write to
the `write_through` generator adds is narrower: a `TeeNode` that survives to execution writes only
if its output is actually pulled.

### Decouple the write with a buffered tee (batchcorder)

Back the consumer with a buffer so the write can run at its own pace, independent of the
downstream consumer, with disk spill when it lags.

**Partially addressed.** `ThreadedBackendWriteThrough` decouples the write via an unbounded
in-memory queue and a background thread. `WritePrimaryWriteThrough` adds the bounded-memory
variant: a `maxsize > 0` queue lets downstream back-pressure the write while still running it on
its own thread (at the cost of being drain-always — see its subsection). What neither ships is
*disk-spill* decoupling: a bounded buffer that spills to disk rather than blocking, which
reintroduces a second puller. batchcorder (ADR-0013, forthcoming) is the candidate, and would
additionally need a durable Parquet writer in its disk layer.

### A dedicated N-way fan-out node

**Deferred.** Phase 1 supports a fan-out of two; N is reached by chaining `TeeNode`s. A native
N-way node can come later if chaining proves insufficient.

## Consequences

### Positive

- A deferred write becomes a first-class side effect that composes with the rest of the graph.
- `TeeNode` hash neutrality means the write composes with caching with no special cases, and a
  downstream cache hit suppresses the write automatically.
- The `WriteThrough` ABC keeps `TeeNode` oblivious to the concrete writer, so new write-throughs drop in by
  subclassing `WriteThrough` without touching the node or the resolution passes.
- `ParquetWriteThrough` provides atomic publish: create via `os.link` (create-or-fail), append via
  merge-then-rename. A partial or aborted run never corrupts the target.
- `.tee()` accepting both `WriteThrough` and `BaseBackend` gives a concise shorthand
  (`t.tee(con, table_name="tgt")`) for the common case, constructing the preferred
  `ThreadedBackendWriteThrough` so the write does not block downstream.
- `drain` defaults to `True`, so the write-through completes the full write even when downstream
  stops early (`LIMIT`/`head`), using a `DrainingIterator` that continues iterating the
  write-through generator in a background thread. No changes to the `WriteThrough` ABC or any
  existing write-through implementation are required — the drain simply continues normal generator
  iteration to exhaustion. Pass `drain=False` to opt into early-stop abort.

### Negative

The first four are the footguns — cases where the write does something a caller would not expect
(or nothing at all). The rest are cost/resource trade-offs.

- **DuckDB (and any single-connection engine) deadlocks.** The streaming tee requires a backend
  whose reader can be pulled concurrently with the outer query. Datafusion and the pandas path
  work; duckdb **deadlocks**, because the tee re-enters the same connection to pull the parent
  while that connection is serving the outer query. Supporting such engines needs the parent
  pulled on a separate connection or thread, which is deferred. This is a hard "feature does not
  work here," not a slowdown.
- **`LIMIT`/`head` writes the whole parent by default.** A downstream early-stop completes the
  write under `drain=True`: the remaining batches are consumed through the write-through in a
  background thread after execution (the thread is non-daemon and is joined before temporary
  tables are dropped, so the parent reader stays valid). A `LIMIT 10` on a billion-row parent
  writes all billion rows — expected given the default, but surprising; pass `drain=False` to
  abort the write on early-stop.
- **The write is not all-or-nothing for backends.** `BackendWriteThrough`'s per-batch path
  (mode-capable backends, e.g. Postgres/ADBC) commits each batch individually, so a mid-stream
  failure leaves earlier batches written; rollback requires the backend's own transactional
  support. Worse, the bulk path (no-mode backends: DataFusion, DuckDB, Pandas) yields every batch
  downstream *before* the single post-stream ingest, so a failure there means downstream already
  consumed data the write-through never persisted — the "write-then-yield" guarantee is inverted.
  Such failures surface as drain/close errors rather than being swallowed, but the divergence is
  real. Only `ParquetWriteThrough` is genuinely atomic.
- **The `drain=False` abort guarantee is transport-dependent.** "Early-stop aborts the write" is a
  property of the client-side generator transport. The shipped `WritePrimaryWriteThrough` and the
  future in-engine transport (e.g. a Postgres data-modifying CTE) are drain-always by construction
  and cannot honor it — callers relying on early-stop suppression must not assume it holds for
  those transports.
- The default backend consumer (`ThreadedBackendWriteThrough`) decouples the write via an
  **unbounded** queue, giving up backpressure: if downstream pulls faster than the write ingests,
  the queue grows without bound. `WritePrimaryWriteThrough(maxsize > 0)` is the shipped
  bounded-memory option (downstream back-pressures the write), at the cost of being drain-always;
  full disk-spill decoupling (batchcorder) is still future. Note the inline `BackendWriteThrough`
  is **not** a bounded-memory fallback in general — only its per-batch path keeps memory at
  O(batch) and back-pressures the producer, at the cost of one ingest roundtrip per batch. Its
  bulk path buffers the entire dataset before a single ingest, so it already sits at worst-case
  memory with no backpressure; against that bulk path the threaded default is a wash-or-win.
- `ParquetWriteThrough` append mode rewrites the entire file on each append (streaming, not
  in-memory, but still O(existing + new) in I/O). A future optimization could append row
  groups without rewriting, but parquet's footer makes that non-trivial.
- A stray `.tmp` or `.merge.tmp` may linger if a consumer abandons the reader without exhausting it (cleanup is
  best-effort; nothing is ever published).
- `BackendWriteThrough.kwargs` is excluded from the identity (hash, equality, token), so any
  option passed through it that changes the rows written — rather than just tuning mechanics —
  silently escapes the build and cache hashes, risking a stale cache hit. The invariant
  ("kwargs tunes mechanics only") is a documented contract, not an enforced one.

## References

- `Read` (deferred read precedent): `python/xorq/expr/relations.py`
- `deferred_read_*`: `python/xorq/common/utils/defer_utils.py`
- `Tag` / `HashingTag`: `python/xorq/expr/relations.py`
- Tag stripping pass: `_remove_non_hashing_tag_nodes` in `python/xorq/expr/api.py`
- Tee stripping pass: `_remove_tee_nodes` in `python/xorq/expr/api.py`
- RemoteTable fan-out (node rewrite at execution): `register_and_transform_remote_tables` in `python/xorq/expr/relations.py`
- Atomic write precedent (temp file + rename): `python/xorq/caching/storage.py`
- ADR-0013: batchcorder StreamCache for RemoteTable fan-out (forthcoming), candidate for the future buffered-tee optimization
- Issue #2087 (build- vs. cache-hash naming): this ADR edits `get_expr_hash` (`python/xorq/common/utils/provenance_utils.py`) and relies on the build/cache hash split. A rename there (`get_expr_hash` → `compute_build_hash`) must update this ADR's references and the `include_tee_nodes` call site.
