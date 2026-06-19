# ADR-0014: TeeNode, deferred write as a side effect

- **Status:** Proposed
- **Date:** 2026-06-10
- **Deciders:** Daniel
- **Context area:** `python/xorq/sinking/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`, `python/xorq/vendor/ibis/expr/types/relations.py`

## Context

Xorq has first-class **deferred reads**: `deferred_read_parquet`, `deferred_read_csv`, and
the `Read` op (`relations.py`) defer a read to execution time, invoking
`getattr(source, method_name)(...)` only when the expression runs. There is no symmetric
**deferred write**. An expression cannot, as part of running, write its rows to a target as
a side effect and keep going.

This ADR adds that: a write performed as a side effect of execution. It is implemented as a
hash-neutral pass-through `TeeNode` that drives a `Sink` generator. A terminal write
node (the eventual terminal `Sink`) is deferred to a later phase (see Alternatives).

## Decision drivers

- A write should be a **side effect** that does not change what an expression evaluates to,
  so it composes with the rest of the graph and with caching.
- The write must **respect the cache**: if downstream work is served from a cache and the
  data is never pulled, the write must not fire.
- "Sink" is **terminal** by convention (data goes in, nothing comes out). The common need
  is the opposite, to write and keep going, so the node that ships first is a pass-through.

## Decision

### One node: a pass-through `TeeNode`

`TeeNode` is a **pass-through**: it evaluates to its parent's rows unchanged and, as those
rows are pulled through it, **hands each batch to a sink** as a side effect. The TeeNode
does not write anything itself; it delegates to a `Sink` whose `sink(batches)`
generator wraps the parent's batch stream. The user-facing `.tee()` builds a `TeeNode`. A
separate terminal write node is deferred (see Alternatives), so `TeeNode` is the only node
this ADR ships.

`TeeNode` does not reference any concrete writer type. It holds a `Sink` in its `sink`
field (see the `Sink` contract below) and is otherwise oblivious to what the sink does
with the batches it receives.

### `TeeNode` is hash-neutral, like `Tag`

`TeeNode` is modeled on `Tag` (`relations.py`). Its schema equals its parent's schema,
and it is **stripped before hashing** by a resolution pass that replaces it with its parent,
exactly as `_remove_non_hashing_tag_nodes` (`api.py`) does for `Tag`. So `expr.tee(s)`
and `expr` produce the **same** content hash, and a `CachedNode` above or below the tee keys
as if the tee were not there.

### The `Sink` contract

A `Sink` is an abstract base class with a single method:

```python
class Sink(abc.ABC):
    @abc.abstractmethod
    def sink(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Pull from batches, write each as a side effect, yield onward."""
        ...
```

`sink(batches)` is a **generator**: it pulls from `batches`, writes each batch as a side
effect, and yields it downstream unchanged. Each sink subclass owns its full lifecycle
(open, commit/publish, abort/cleanup) inside its `sink` implementation. The downstream
consumer drives iteration — the sink never independently pulls.

This gives the cache-respecting behavior for free:

- **Single puller.** The only consumer of the stream is downstream. The sink never pulls
  independently; it receives batches the downstream pull already produced. If downstream
  does not pull (a cache hit), the generator never runs and nothing is written.
- **Write-then-yield.** For sinks that commit incrementally (`ParquetSink`, and
  `BackendSink` against a backend whose `read_record_batches` accepts a write `mode`),
  every batch handed downstream was written by the sink first, so the written set equals
  the delivered set with no off-by-one. Bulk backends are the exception: a backend that
  cannot append must ingest in a single call, so `BackendSink._sink_bulk` yields every
  batch downstream first and ingests once after the stream is exhausted. There the write
  trails delivery — a post-stream ingest failure means downstream already saw data the
  sink never persisted. Such failures surface as drain/close errors rather than being
  swallowed.
- **Default: lock-step.** By default the write sits in the pull path, so a slow write
  blocks the producer. `ThreadedBackendSink` relaxes this by running the ingest on a
  background thread with an unbounded queue, decoupling the write from the downstream
  consumer at the cost of buffering lag in memory.

### `ParquetSink`: single-file parquet consumer

`ParquetSink(path, mode)` writes to a single, user-named parquet file. `path` is the final
file path, known at construction time. `sink` stages batches to a temp sibling
(`path + ".tmp"`), then publishes atomically after clean exhaustion:

- **`create`**: publishes via `os.link(tmp, path)` — atomic create-or-fail. If `path`
  already exists the link raises `FileExistsError`; no glob, no lock needed. The temp is
  always cleaned up.
- **`append`**: acquires an `fcntl.flock` on `path + ".lock"`, then streams the existing
  file's row groups followed by the staged batches through a `ParquetWriter` into a second
  temp (`path + ".merge.tmp"`), and renames that over the target. Both reads use
  `ParquetFile.iter_batches()`, so memory stays at O(batch_size) regardless of existing file
  size. The lock serializes concurrent appenders; the rename is atomic. The lock file is
  removed after the operation.

An error or early stop mid-stream discards the temp file and publishes nothing. Object
stores have no atomic link/rename and need a different publish; that is out of scope for
Phase 1.

### `BackendSink`: per-backend ingest consumer

`BackendSink(con, table_name, mode, kwargs)` delegates writes to any xorq backend's
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

`ThreadedBackendSink` is a `BackendSink` subclass that replaces both the per-batch
round-trips and the bulk buffer with a single streaming `read_record_batches` call on a
background thread. A `queue.SimpleQueue` bridges the generator and the thread: each batch
is pushed onto the queue and yielded downstream; the thread's reader drains the queue. The
queue is unbounded (no backpressure) — a bounded queue would deadlock if the sink thread
died without draining. Rollback on error follows the backend's own semantics.

### `.tee()`: the user-facing method

`.tee()` is polymorphic — it accepts either a `Sink` directly or a backend connection:

```python
def tee(self, target: Sink | BaseBackend, *, table_name=None, drain=False, **kwargs) -> Table
```

When `target` is a `BaseBackend` (with `read_record_batches`), `.tee()` auto-creates a
`BackendSink` from `table_name` and the remaining `**kwargs`. When `target` is already a
`Sink`, extra keyword arguments are rejected.

### `drain=True`: completing the write on early stop

By default, a downstream early-stop (`LIMIT`/`head`) aborts the sink — the generator sees
`GeneratorExit`, cleanup runs, and nothing is published. With `drain=True`, the execution
pipeline wraps the sink generator in a `DrainingIterator` (`python/xorq/sinking/sink.py`).

During normal iteration, `DrainingIterator` is a transparent pass-through. After downstream
execution completes, the pipeline calls `close()` on each `DrainingIterator`. If the
generator was not fully consumed, `close()` spawns a non-daemon background thread that
continues iterating the generator — each `next()` triggers the sink's write side-effect as
usual, but the yielded batches are discarded since no downstream consumer remains. The
pipeline then calls `join()` to wait for the drain thread before dropping temporary tables
(so the parent reader stays valid during the drain).

The key invariant that makes this safe: the "no independent pulling" constraint exists to
avoid buffering for downstream. Once downstream is done, there is no downstream to buffer
for, so the sink can freely consume the remainder of the parent stream.

If the drain thread encounters an error, `join()` re-raises it to the caller. The lifecycle
is explicit — there is no `__del__` finalizer, since the backend holds the reader alive
past GC, making finalizer-based cleanup unreliable.

No changes to the `Sink` ABC or any existing sink implementation are required.

### Fan-out of two, chained for N

Phase 1 supports a fan-out of two: one main consumer plus one inline write. A fan-out of N
is achieved by **chaining N-1 `TeeNode`s**, not by a dedicated N-way node.

### Durability and relationship to `into_backend`

A deferred write yields a **persistent, user-named, user-owned** target. It is never temporary
and never auto-dropped; teardown is explicit. This is the concrete difference from
`RemoteTable`/`into_backend`, which produces an **anonymous, ephemeral** table dropped in
`clean_up` (`api.py`). They share the underlying write transport but are separate
constructs, and the API keeps them separate.

### Node shape

```python
# python/xorq/expr/relations.py

class TeeNode(ops.Relation):
    schema: Schema            # == parent.schema; pass-through
    parent: ops.Relation
    sink: Sink                # a Sink whose sink() drives the write
    drain: bool = False       # drain remaining batches on early stop
    values = FrozenDict()     # hash-neutral, like Tag
```

`TeeNode` declares its identity via `__dasher_tokenize__`, which returns
`("tee-node", self.schema, self.sink)` — delegating sink identity to each
`Sink` subclass's own `__dasher_tokenize__`.  `drain` is excluded from the
token because it is purely an execution-time concern that does not change the
logical result.  The cache hash path strips `TeeNode` (like `Tag`), so
`__dasher_tokenize__` is only reached when `_hash_expr_components` explicitly
tokenizes the extracted nodes.  The build hash path (`get_expr_hash`) sets
`_include_tee_nodes` so that different sinks produce different build artifacts.

```python
# python/xorq/sinking/sink.py

class ParquetSink(Sink):
    path: Path                # the final file path (not a directory)
    mode: SinkMode            # "create" (link) | "append" (merge + rename)
    def __dasher_tokenize__(self): return ("ParquetSink", str(self.path), self.mode)
    def sink(self, batches): ...  # stage to path.tmp, publish on exhaustion

class BackendSink(Sink):
    con: Any
    table_name: str
    mode: SinkMode
    kwargs: dict
    def __dasher_tokenize__(self): return ("BackendSink", getattr(self.con, "name", ""), self.table_name, self.mode)
    def sink(self, batches): ...  # per-batch or bulk, depending on backend
```

```python
# Table.tee (vendored relations.py): the user-facing attach

def tee(self, target: Sink | BaseBackend, *, table_name=None, drain=False, **kwargs) -> Table:
    ...
    op = TeeNode(schema=self.schema(), parent=self.op(), sink=sink, drain=drain)
    return op.to_expr()
```

Two resolution passes keep hash neutrality and the side effect separate:

- **Hashing**: a strip pass replaces each `TeeNode` with its parent before the hash is
  computed, like `_remove_non_hashing_tag_nodes` for `Tag`. `_remove_tee_nodes`
  does the same off the SQL path.
- **Execution**: `register_and_transform_tee_nodes` replaces each surviving `TeeNode` with a
  backend table fed by `sink.sink(reader)`:

  ```python
  reader = parent_expr.to_pyarrow_batches()
  wrapped = pa.RecordBatchReader.from_batches(reader.schema, node.sink.sink(reader))
  table = con.read_record_batches(wrapped, table_name=gen_name())
  ```

  This mirrors the RemoteTable pass (`register_and_transform_remote_tables`). It runs after
  cache resolution, so a downstream cache hit prunes the tee before this pass sees it and the
  write never fires.

### Naming

The user-facing method is `.tee()`, matching the Unix `tee` command: data flows through and
a copy is written as a side effect. The consumer abstraction is named `Sink`, and the
field on `TeeNode` is `sink`. This split is deliberate: "tee" names the pass-through
behavior the caller sees, while "sink" names what the consumer does with the data it
receives (writes it somewhere). A future terminal write node (see Alternatives) would use
`Sink` directly without wrapping it in a `TeeNode`.

## Alternatives considered

### A terminal `Sink` (deferred write via `methodcaller`)

A terminal node, the true counterpart to `deferred_read_*`: data in, nothing past. Executing
it would fire `operator.methodcaller(sink_method)(parent)` at execution time, mirroring how
`Read.make_dt` resolves a deferred read via `getattr(source, method_name)(...)`.

**Deferred.** It is orthogonal to `TeeNode` (neither would reference the other) and not needed
for the pass-through write that ships first. It is recorded here as the next node to add when
a terminal, non-pass-through write is wanted.

### A single node that both writes and passes through

One node that is simultaneously a terminal sink and a pass-through tee.

**Rejected.** It conflates two behaviors with opposite modes. A pass-through `TeeNode` and a
(deferred) terminal `Sink` stay orthogonal and each name matches its behavior.

### Write unconditionally at execution, regardless of pull

Fire the write whenever a write node is present in the executed subgraph.

**Rejected.** It would fire on a pure cache hit, doing I/O for data nobody pulled. Tying the
write to the `sink` generator, which only runs when the main consumer pulls, makes "cache
hit means no write" follow automatically.

### Decouple the write with a buffered tee (batchcorder)

Back the consumer with a buffer so the write can run at its own pace, independent of the
downstream consumer, with disk spill when it lags.

**Partially addressed.** `ThreadedBackendSink` decouples the write via an unbounded
in-memory queue and a background thread. Full disk-spill decoupling needs a bounded buffer
that spills to disk, which reintroduces a second puller. batchcorder (ADR-0013, forthcoming)
is the candidate, and would additionally need a durable Parquet sink writer in its disk
layer.

### A dedicated N-way fan-out node

**Deferred.** Phase 1 supports a fan-out of two; N is reached by chaining `TeeNode`s. A native
N-way node can come later if chaining proves insufficient.

## Consequences

### Positive

- A deferred write becomes a first-class side effect that composes with the rest of the graph.
- `TeeNode` hash neutrality means the write composes with caching with no special cases, and a
  downstream cache hit suppresses the write automatically.
- The `Sink` ABC keeps `TeeNode` oblivious to the writer, so new sinks drop in by
  subclassing `Sink` without touching the node or the resolution passes.
- `ParquetSink` provides atomic publish: create via `os.link` (create-or-fail), append via
  merge-then-rename. A partial or aborted run never corrupts the target.
- `.tee()` accepting both `Sink` and `BaseBackend` gives a concise shorthand
  (`t.tee(con, table_name="tgt")`) for the common case.
- `drain=True` lets the sink complete the full write even when downstream stops early
  (`LIMIT`/`head`), using a `DrainingIterator` that continues iterating the sink generator
  in a background thread. No changes to the `Sink` ABC or any existing sink implementation
  are required — the drain simply continues normal generator iteration to exhaustion.

### Negative

- The default inline generator runs the write and the downstream consumer lock-step, so a
  slow write slows downstream. `ThreadedBackendSink` decouples via an unbounded queue, but
  full disk-spill decoupling (batchcorder) is a future optimization.
- A downstream early-stop (`LIMIT`/`head`) publishes nothing by default. With `drain=True`,
  the remaining batches are consumed through the sink in a background thread after execution,
  so the write completes. The drain thread is non-daemon and is joined before temporary
  tables are dropped, so the parent reader stays valid. A `LIMIT 10` on a billion-row
  parent with `drain=True` writes all billion rows — expected, but potentially surprising.
- `ParquetSink` append mode rewrites the entire file on each append (streaming, not
  in-memory, but still O(existing + new) in I/O). A future optimization could append row
  groups without rewriting, but parquet's footer makes that non-trivial.
- A stray `.tmp` or `.merge.tmp` may linger if a consumer abandons the reader without exhausting it (cleanup is
  best-effort; nothing is ever published).
- The streaming tee requires a backend whose reader can be pulled concurrently with the outer
  query. Datafusion and the pandas path work; a single-connection engine like duckdb
  **deadlocks**, because the tee re-enters the same connection to pull the parent while that
  connection is serving the outer query. Supporting such engines needs the parent pulled on a
  separate connection or thread, which is deferred.
- `BackendSink`'s per-batch path (for backends with `mode` support) commits each batch
  individually, so a mid-stream failure leaves earlier batches written. This is not
  all-or-nothing like `ParquetSink`; rollback requires the backend's own transactional support.

## References

- `Read` (deferred read precedent): `python/xorq/expr/relations.py`
- `deferred_read_*`: `python/xorq/common/utils/defer_utils.py`
- `Tag` / `HashingTag`: `python/xorq/expr/relations.py`
- Tag stripping pass: `_remove_non_hashing_tag_nodes` in `python/xorq/expr/api.py`
- Tee stripping pass: `_remove_tee_nodes` in `python/xorq/expr/api.py`
- RemoteTable fan-out (node rewrite at execution): `register_and_transform_remote_tables` in `python/xorq/expr/relations.py`
- Atomic write precedent (temp file + rename): `python/xorq/caching/storage.py`
- ADR-0013: batchcorder StreamCache for RemoteTable fan-out (forthcoming), candidate for the future buffered-tee optimization
