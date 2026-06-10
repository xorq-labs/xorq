# ADR-0014: TeeNode, deferred write as a side effect

- **Status:** Proposed
- **Date:** 2026-06-10
- **Deciders:** Daniel
- **Context area:** `python/xorq/sinking/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`

## Context

Xorq has first-class **deferred reads**: `deferred_read_parquet`, `deferred_read_csv`, and
the `Read` op (`relations.py:564`) defer a read to execution time, invoking
`getattr(source, method_name)(...)` only when the expression runs. There is no symmetric
**deferred write**. An expression cannot, as part of running, write its rows to a target as
a side effect and keep going.

This ADR adds that: a write performed as a side effect of execution. It is implemented as a
hash-neutral pass-through `TeeNode` that drives a generic RecordBatch consumer. A terminal
write node (the eventual `SinkNode`) is deferred to a later phase (see Alternatives).

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
rows are pulled through it, **hands each batch to a consumer** as a side effect. The TeeNode
does not write anything itself; it passes the batch to the consumer, and the consumer (here
`ParquetSink`) writes it to the destination. The user-facing `.sink()` builds a `TeeNode`. A
separate terminal write node is deferred (see Alternatives), so `TeeNode` is the only node
this ADR ships.

`TeeNode` does not reference any write node or any concrete writer type. It holds a generic
**consumer** in its `tee` field (see the consumer contract below) and is otherwise oblivious
to what the consumer does with the batches it receives.

### `TeeNode` is hash-neutral, like `Tag`

`TeeNode` is modeled on `Tag` (`relations.py:101`). Its schema equals its parent's schema,
and it is **stripped before hashing** by a resolution pass that replaces it with its parent,
exactly as `_remove_non_hashing_tag_nodes` (`api.py:347`) does for `Tag`. So `expr.sink(s)`
and `expr` produce the **same** content hash, and a `CachedNode` above or below the tee keys
as if the tee were not there.

### The write is an inline pass-through generator

At execution the tee is driven by an inline generator over the parent's batches: for each
batch it pushes to the consumer, then yields downstream.

```python
def _drive_tee(reader, consumer):
    try:
        for batch in reader:
            consumer.read(batch)   # push-fed: receives an already-pulled batch
            yield batch            # pass-through
        consumer.commit()
    except BaseException:
        consumer.abort()
        raise
```

This gives the cache-respecting behavior for free:

- **Single puller.** The only consumer of the stream is downstream. The writer never pulls;
  it receives batches the downstream pull already produced. If downstream does not pull (a
  cache hit), the generator never runs and nothing is written.
- **write-then-yield.** Every batch handed downstream was pushed to the consumer first, so
  the written set equals the delivered set with no off-by-one.
- **True tee, lock-step.** The write sits in the pull path, so a slow write blocks the
  producer. There is no buffer, so nothing can overflow. The cost is that the write and the
  downstream consumer run at the same pace; they are not decoupled.

### The consumer contract

`tee` is any object implementing a small RecordBatch-consumer protocol:

- **`read(batch)`** receive one `pyarrow.RecordBatch`.
- **`commit()`** publish; called once on full exhaustion.
- **`abort()`** discard; called on early stop or error.

The consumer owns its own durability. Nothing is published until `commit`, so a cache hit, a
downstream `LIMIT`, or an error publishes nothing and leaves any prior target intact (every
run is all-or-nothing).

### `ParquetSink`: the Phase-1 consumer

`ParquetSink(path, mode)` stages batches to a temp file on `read`, and on `commit` renames it
into the target directory (the standard atomic-write idiom). The two modes are plain write
behavior:

- **`append`**: `commit` adds a new file to the directory; existing files are untouched. A
  single rename, genuinely atomic.
- **`create`**: refuses to overwrite. It **raises** if the target already has parquet files
  (like SQL `CREATE TABLE`, not `CREATE OR REPLACE`); otherwise `commit` adds the file.

The temp lives on the same filesystem as the target so the rename is atomic. Object stores
have no atomic rename and need a different publish; that is out of scope for Phase 1.

### Fan-out of two, chained for N

Phase 1 supports a fan-out of two: one main consumer plus one inline write. A fan-out of N
is achieved by **chaining N-1 `TeeNode`s**, not by a dedicated N-way node.

### Durability and relationship to `into_backend`

A deferred write yields a **persistent, user-named, user-owned** target. It is never temporary
and never auto-dropped; teardown is explicit. This is the concrete difference from
`RemoteTable`/`into_backend`, which produces an **anonymous, ephemeral** table dropped in
`clean_up` (`api.py:482`). They share the underlying write transport but are separate
constructs, and the API keeps them separate.

### Node shape

```python
# python/xorq/expr/relations.py

class TeeNode(ops.Relation):
    schema: Schema            # == parent.schema; pass-through
    parent: ops.Relation
    tee: Any = None           # a RecordBatch consumer (read / commit / abort)
    values = FrozenDict()     # hash-neutral, like Tag

    def __dasher_tokenize__(self):
        return ("normalize_tee_node", self.schema, self.parent)
```

```python
# python/xorq/sinking/__init__.py

class ParquetSink:
    def read(self, batch): ...    # stage to a temp file
    def commit(self): ...         # rename temp into the target dir (append/create)
    def abort(self): ...          # discard the temp
```

```python
# Table.sink (vendored relations.py): the user-facing attach

def sink(self, sink):
    op = TeeNode(schema=self.schema(), parent=self.op(), tee=sink)
    return op.to_expr()
```

Two resolution passes keep hash neutrality and the side effect separate:

- **Hashing**: a strip pass replaces each `TeeNode` with its parent before the hash is
  computed, like `_remove_non_hashing_tag_nodes` (`api.py:347`) for `Tag`. `_remove_tee_nodes`
  does the same off the SQL path.
- **Execution**: `register_and_transform_tee_nodes` replaces each surviving `TeeNode` with a
  backend table fed by `_drive_tee`, mirroring the RemoteTable pass
  (`register_and_transform_remote_tables`, `relations.py:587`). It runs after cache
  resolution, so a downstream cache hit prunes the tee before this pass and the write never
  fires.

### Naming

The user-facing function is `.sink()` even though it builds a pass-through `TeeNode`. This
bends the terminal convention of "sink" on purpose, and the docs must say so up front:
`.sink()` writes as a side effect and the pipeline keeps going.

## Alternatives considered

### A terminal `SinkNode` (deferred write via `methodcaller`)

A terminal node, the true counterpart to `deferred_read_*`: data in, nothing past. Executing
it would fire `operator.methodcaller(sink_method)(parent)` at execution time, mirroring how
`Read.make_dt` resolves a deferred read via `getattr(source, method_name)(...)`.

**Deferred.** It is orthogonal to `TeeNode` (neither would reference the other) and not needed
for the pass-through write that ships first. It is recorded here as the next node to add when
a terminal, non-pass-through write is wanted.

### A single node that both writes and passes through

One node that is simultaneously a terminal sink and a pass-through tee.

**Rejected.** It conflates two behaviors with opposite modes. A pass-through `TeeNode` and a
(deferred) terminal `SinkNode` stay orthogonal and each name matches its behavior.

### Write unconditionally at execution, regardless of pull

Fire the write whenever a write node is present in the executed subgraph.

**Rejected.** It would fire on a pure cache hit, doing I/O for data nobody pulled. Tying the
write to the inline generator, which only runs when the main consumer pulls, makes "cache hit
means no write" follow automatically.

### Decouple the write with a buffered tee (batchcorder)

Back the consumer with a buffer so the write can run at its own pace, independent of the
downstream consumer, with disk spill when it lags.

**Deferred, future optimization.** The inline generator is lock-step: a slow write slows
downstream. Decoupling them needs a bounded buffer that spills to disk, which reintroduces a
second puller. batchcorder (ADR-0013, forthcoming) is the candidate, and would additionally
need a durable Parquet sink writer in its disk layer. Not required for Phase 1.

### A dedicated N-way fan-out node

**Deferred.** Phase 1 supports a fan-out of two; N is reached by chaining `TeeNode`s. A native
N-way node can come later if chaining proves insufficient.

## Consequences

### Positive

- A deferred write becomes a first-class side effect that composes with the rest of the graph.
- `TeeNode` hash neutrality means the write composes with caching with no special cases, and a
  downstream cache hit suppresses the write automatically.
- The generic `read`/`commit`/`abort` consumer keeps `TeeNode` oblivious to the writer, so new
  sinks (database, iceberg) drop in without touching the node.
- Atomic, all-or-nothing publish: a partial or aborted run never corrupts the target.

### Negative

- `.sink()` building a pass-through rather than a terminal node bends the "sink" name; docs
  must make this explicit or users will expect `.sink()` to end the stream.
- The inline generator runs the write and the downstream consumer lock-step, so a slow write
  slows downstream. Decoupling them (batchcorder) is a future optimization.
- A downstream early-stop (`LIMIT`/`head`) publishes nothing rather than the pulled prefix.
- A stray `.tmp` may linger if a consumer abandons the reader without exhausting it (cleanup is
  best-effort; nothing is ever published).
- The streaming tee requires a backend whose reader can be pulled concurrently with the outer
  query. Datafusion and the pandas path work; a single-connection engine like duckdb
  **deadlocks**, because the tee re-enters the same connection to pull the parent while that
  connection is serving the outer query. Supporting such engines needs the parent pulled on a
  separate connection or thread, which is deferred.

## References

- `Read` (deferred read precedent): `python/xorq/expr/relations.py:564`
- `deferred_read_*`: `python/xorq/common/utils/defer_utils.py`
- `Tag` / `HashingTag`: `python/xorq/expr/relations.py:101`
- Tag stripping pass: `_remove_non_hashing_tag_nodes`, `python/xorq/expr/api.py:347`
- RemoteTable fan-out (node rewrite at execution): `register_and_transform_remote_tables`, `python/xorq/expr/relations.py:587`
- Atomic write precedent (temp file + rename): `python/xorq/caching/storage.py:80`
- ADR-0013: batchcorder StreamCache for RemoteTable fan-out (forthcoming), candidate for the future buffered-tee optimization
