# ADR-0014: SinkNode and TeeNode, deferred write as a side effect

- **Status:** Proposed
- **Date:** 2026-06-10
- **Deciders:** Daniel
- **Context area:** `python/xorq/sinking/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`

## Context

Xorq has first-class **deferred reads**: `deferred_read_parquet`, `deferred_read_csv`, and
the `Read` op (`relations.py:564`) defer a read to execution time, invoking
`getattr(source, method_name)(...)` only when the expression runs. There is no symmetric
**deferred write**. An expression cannot, as part of running, write its rows to a target as
a side effect.

This ADR adds that: a write performed as a side effect of execution, the counterpart to
`deferred_read_*`. The write is implemented as an **inline pass-through generator**: as each
batch flows to the downstream consumer, it is written first. Backpressure needs no special
machinery, because there is a single puller (the downstream consumer) and the write is
push-fed on its pull path.

## Decision drivers

- A write should be a **side effect** that does not change what an expression evaluates to,
  so it composes with the rest of the graph and with caching.
- The write must **respect the cache**: if downstream work is served from a cache and the
  data is never pulled, the write must not fire.
- "Sink" is **terminal** by connotation (data goes in, nothing comes out). The common need
  is the opposite, to write and keep going, so the two behaviors need separate nodes.

## Decision

### Two nodes: a terminal `SinkNode` and a pass-through `TeeNode`

The capability is split across two independent node types.

- **`SinkNode`** is a **terminal** deferred write. Data flows in; nothing flows past it.
- **`TeeNode`** is a **pass-through**. It forks its parent's stream into a main consumer
  that continues downstream and a side-effect consumer that performs a write. It evaluates
  to its parent's rows.

The two are independent; users never wire them together by hand. The user-facing `.sink()`
function composes them: it builds a `TeeNode` whose side-effect leg terminates in a
`SinkNode`. A bare terminal `SinkNode` is an internal or edge construct.

### `SinkNode` is a deferred write, the counterpart to `deferred_read_*`

`SinkNode` mirrors `Read` (`relations.py:564`). Where `Read` carries a `method_name` and
resolves by calling `getattr(source, method_name)(...)`, a `SinkNode` carries a
`sink_method` and resolves at execution time by calling

```python
operator.methodcaller(sink_op.sink_method)(sink_op.parent)
```

It is terminal: it produces no downstream relation, so nothing can read through a sink.
That is the connotation the name carries on purpose.

### Write semantics: `create` and `append`

A sink writes in one of two ways, framed purely as write behavior:

- **`create`**: the target is (re)created from the written batches, replacing any prior
  contents.
- **`append`**: the written batches are added to the existing target.

This is the full surface: replace or add.

### `TeeNode` writes as a side effect and passes the stream through

`TeeNode` is transparent, modeled on `Tag` (`relations.py:101`). Its schema equals its
parent's schema, it evaluates to its parent's rows unchanged, and it is **stripped before
hashing** by a resolution pass that replaces it with its parent, exactly as
`_remove_non_hashing_tag_nodes` (`api.py:347`) does for `Tag`. So `expr.sink(s)` and `expr`
produce the **same** content hash.

The write is the side-effect leg, and it fires only when the main consumer pulls batches
through the tee. A direct consequence: on a **downstream cache hit** the cached value is
returned without pulling through the tee, so the write never fires.

### The write is an inline pass-through generator

The `TeeNode`'s side-effect leg is an inline generator over the parent's batches: for each
batch it **writes, then yields** downstream.

```python
def _tee_write(batches, writer):
    for batch in batches:
        writer.write_batch(batch)   # side effect (push-fed)
        yield batch                 # pass-through
```

This gives the cache-respecting behavior for free:

- **Single puller.** The only consumer is downstream. The writer is not an iterator and
  never pulls; it receives batches the downstream pull already produced. If downstream does
  not pull (a cache hit), the generator never runs and nothing is written.
- **write-then-yield.** Every batch handed downstream was written first, so the written set
  equals the delivered set with no off-by-one. A downstream that stops early wrote exactly
  the prefix it pulled.
- **True tee, lock-step.** The write sits in the pull path, so a slow write blocks the
  producer. There is no buffer, so nothing can overflow. The cost is that the write and the
  downstream consumer run at the same pace; they are not decoupled.

### Atomic publish: stage, then move on exhaustion

The generator writes to a **staged temp location** and moves it to the target **only if the
iterator is exhausted** (the run completed). Any other exit (cache hit, downstream `LIMIT`,
error) publishes nothing and leaves the prior target intact, so every run is all-or-nothing.
This is the standard atomic-write idiom (write to temp, then `rename`), and it maps onto the
two write modes:

- **`append`**: `rename` the staged file into the target directory as a new file. A single
  rename, genuinely atomic; existing files are untouched.
- **`create`**: swap the staged output into the target name, replacing the old contents only
  on success.

The temp must live on the same filesystem as the target for the rename to be atomic. Object
stores have no atomic rename and need a different publish; that is out of scope for Phase 1.

### Fan-out of two, chained for N

Phase 1 supports a fan-out of two: one main consumer plus one inline write. A fan-out of N
is achieved by **chaining N-1 `TeeNode`s**, not by a dedicated N-way node.

### Durability and relationship to `into_backend`

A deferred write yields a **durable, user-named, user-owned** target. It is never temporary
and never auto-dropped; teardown is explicit. This is the concrete difference from
`RemoteTable`/`into_backend`, which produces an **anonymous, ephemeral** table dropped in
`clean_up` (`api.py:482`). They share the underlying write transport but are separate
constructs, and the API keeps them separate.

### Node shape

A sketch, not a contract; field names settle in Phase 1. Both nodes live in
`python/xorq/expr/relations.py` next to `Read`, `Tag`, and `CachedNode`.

```python
# python/xorq/expr/relations.py

class SinkNode(ops.Relation):
    parent: ops.Relation
    sink_method: str          # method invoked on the parent at execution time
    # terminal: produces no downstream relation

    def write(self):
        # fires the deferred write as a side effect; mirrors Read.make_dt
        return operator.methodcaller(self.sink_method)(self.parent.to_expr())


class TeeNode(ops.Relation):
    parent: ops.Relation
    schema: Schema            # == parent.schema; pass-through
    sink: SinkNode            # the side-effect leg's terminal write
    values = FrozenDict()     # transparent, like Tag

    def __dasher_tokenize__(self):
        # contributes nothing beyond the parent; stripped before hashing
        return ("normalize_tee_node", self.parent)
```

```python
# python/xorq/expr/api.py : the user-facing attach

def sink(expr, storage, *, mode="append", **kwargs):
    parent = expr.op()
    sink_node = SinkNode(parent=parent, sink_method=...)
    return TeeNode(parent=parent, schema=parent.schema, sink=sink_node).to_expr()
```

Two resolution passes keep transparency and the side effect separate:

- **Hashing**: a strip pass replaces each `TeeNode` with its parent before the hash is
  computed, like `_remove_non_hashing_tag_nodes` (`api.py:347`) for `Tag`.
- **Execution**: a transform pass replaces each `TeeNode` with the inline pass-through
  generator above over the parent's batches, mirroring how the RemoteTable pass rewrites
  nodes at execution time (`register_and_transform_remote_tables`, `relations.py:587`). The
  generator stages the write and moves it to the target on exhaustion; the downstream
  consumer is the only puller. A terminal `SinkNode` is the same generator drained without
  yielding.

### Naming

The user-facing function stays `.sink()` even though it builds a pass-through `TeeNode`, not
a terminal `SinkNode`. This bends the terminal connotation of "sink" on purpose, and the
documentation must say so up front: `.sink()` writes as a side effect and the pipeline keeps
going. The internal node names match their behavior: `SinkNode` is the terminal write,
`TeeNode` is the pass-through.

## Alternatives considered

### A single node that both writes and passes through

One node that is simultaneously a terminal "sink" and a pass-through tee.

**Rejected.** It conflates two behaviors with opposite shapes. "Sink" connotes terminal
(nothing flows past), while the common need is to write and continue. Splitting into a
terminal `SinkNode` and a pass-through `TeeNode` keeps each name honest and lets `.sink()`
compose them.

### Write unconditionally at execution, regardless of pull

Fire the write whenever a write node is present in the executed subgraph.

**Rejected.** It would fire on a pure cache hit, doing I/O for data nobody pulled. Tying the
write to the side-effect leg of the tee, which only runs when the main consumer pulls, makes
"cache hit means no write" fall out for free.

### A dedicated N-way fan-out node

Provide one node that fans the stream out to N consumers directly.

**Deferred.** Phase 1 supports a fan-out of two; N is reached by chaining `TeeNode`s. A
native N-way node can come later if chaining proves insufficient.

### Decouple the write with a buffered tee (batchcorder)

Back the side-effect leg with a buffer so the write can run at its own pace, independent of
the downstream consumer, with disk spill when it lags.

**Deferred, future optimization.** The inline generator is lock-step: a slow write slows
downstream. Decoupling them needs a bounded buffer that spills to disk, which reintroduces a second
puller plus machinery to keep the write tied to the main consumer. batchcorder (ADR-0013,
forthcoming) is the candidate, and would additionally need a durable Parquet sink writer in
its disk layer, separate from its ephemeral IPC replay store. This is a throughput
optimization, not required for Phase 1.

## Consequences

### Positive

- A deferred write becomes the symmetric counterpart to `deferred_read_*`.
- Splitting terminal `SinkNode` from pass-through `TeeNode` keeps the "sink is terminal"
  connotation intact while still letting a pipeline write and continue.
- TeeNode transparency means the write composes with caching with no special cases, and a
  downstream cache hit suppresses the write automatically.

### Negative

- `.sink()` building a pass-through rather than a terminal node bends the "sink" name; docs
  must state this loudly or users will expect `.sink()` to end the stream.
- The inline generator runs the write and the downstream consumer lock-step, so a slow
  write slows downstream. Decoupling them (batchcorder) is a future optimization.
- Atomic publish is all-or-nothing per run: a downstream early-stop (`LIMIT`/`head`)
  publishes nothing rather than the pulled prefix.

## References

- ADR-0013: batchcorder StreamCache for RemoteTable fan-out (forthcoming), candidate for the future buffered-tee optimization
- `Read` (deferred read precedent): `python/xorq/expr/relations.py:564`
- `deferred_read_*`: `python/xorq/common/utils/defer_utils.py`
- `Tag` / `HashingTag`: `python/xorq/expr/relations.py:101`
- Tag stripping pass: `_remove_non_hashing_tag_nodes`, `python/xorq/expr/api.py:347`
- RemoteTable fan-out (node rewrite at execution): `register_and_transform_remote_tables`, `python/xorq/expr/relations.py:587`
- Atomic write precedent (temp file + rename): `python/xorq/caching/storage.py:80`
