# ADR-0014: SinkNode and TeeNode, deferred write as a side effect

- **Status:** Proposed
- **Date:** 2026-06-10
- **Deciders:** Daniel
- **Context area:** `python/xorq/sinking/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`

## Context

Xorq has first-class **deferred reads**: `deferred_read_parquet`, `deferred_read_csv`,
and the `Read` op (`relations.py:564`) defer a read to execution time, invoking
`getattr(source, method_name)(...)` only when the expression runs. There is no symmetric
**deferred write**. An expression cannot, as part of running, write its rows to a durable
target as a side effect and then carry on.

This ADR adds that capability: a write that happens as a side effect of execution, the
counterpart to `deferred_read_*`. The design is framed entirely around two things, the
**deferred write** itself and the **backpressure** between the write and the rest of the
pipeline.

## Decision drivers

- A write should be expressible as a **side effect** that does not change what an
  expression evaluates to, so it composes with the rest of the graph and with caching.
- The write must **respect the cache**: if downstream work is served from a cache and the
  data is never pulled, the write must not fire.
- "Sink" is **terminal** by connotation (data goes in, nothing comes out). The common need
  is the opposite, to write and keep going, so the two behaviors need separate nodes.

## Decision

### Two nodes: a terminal `SinkNode` and a passthrough `TeeNode`

The capability is split across two independent node types.

- **`SinkNode`** is a **terminal** deferred write. Data flows in; nothing flows past it.
- **`TeeNode`** is a **passthrough**. It forks its parent's stream into two consumers: a
  main consumer that continues downstream, and a side-effect consumer that performs a
  write. It evaluates to its parent's rows.

The two nodes are independent; users never wire them together by hand. The user-facing
`.sink()` function composes them: it builds a `TeeNode` whose side-effect leg terminates
in a `SinkNode`. A bare terminal `SinkNode` is an internal or edge construct, not the
common path.

### `SinkNode` is a deferred write, the counterpart to `deferred_read_*`

`SinkNode` mirrors `Read` (`relations.py:564`). Where `Read` carries a `method_name` and
read kwargs and resolves by calling `getattr(source, method_name)(...)`, a `SinkNode`
carries a `sink_method` and write kwargs and resolves by calling

```python
operator.methodcaller(sink_op.sink_method, **kwargs)(sink_op.parent)
```

at execution time. It is terminal: it produces no downstream relation, so nothing can read
"through" a sink. This is the connotation the name carries on purpose.

### `TeeNode` writes as a side effect and passes the stream through

`TeeNode` is transparent, modeled on `Tag` (`relations.py:101`). Its schema equals its
parent's schema, it evaluates to its parent's rows unchanged, and it is **stripped before
hashing**: a resolution pass replaces it with its parent before the content hash is
computed, exactly as `_remove_non_hashing_tag_nodes` (`api.py:347`) does for `Tag`. So
`expr.sink(s)` and `expr` produce the **same** content hash, and the node is invisible to
identity.

The write is the side-effect leg. When the main consumer pulls batches through the tee,
the same batches are driven down the side-effect leg into the `SinkNode`, so the write
fires as a consequence of the pipeline being pulled, never on its own.

### Backpressure: a true tee is the target, batchcorder is the Phase-1 mechanism

Two backpressure shapes matter here:

- A **tee** applies backpressure from its **slowest** consumer. Whichever consumer cannot
  keep up is what stops the producer from producing more batches.
- A **batchcorder** is the inverse: backpressure comes from the **fastest** consumer. The
  producer runs as fast as the fastest consumer, and the delta between the fastest and the
  slowest consumer is buffered.

The **target** behavior of `TeeNode` is a true tee: if the write consumer falls behind, it
blocks the producer rather than letting an unbounded buffer grow. In Phase 1 the
side-effect leg is implemented with **batchcorder** (a buffering node of cardinality one
whose evicted batches feed a `RecordBatchReader`, which in turn feeds the `SinkNode`). With
batchcorder, the write consumer is allowed to lag and the delta is buffered. We expect
side-effect consumers (writes) to **never** apply backpressure in practice; the goal is
that when one does, the tee blocks instead of failing. ADR-0014 therefore has a **hard
dependency on batchcorder (ADR-0013, forthcoming)** and is sequenced after it.

Phase 1 supports a fan-out of two: one backpressure-capable main consumer plus one
side-effect consumer. A fan-out of N is achieved by **chaining N-1 `TeeNode`s**, not by a
dedicated N-way node.

### Composition with caching

Because `TeeNode` is hash-transparent, a `CachedNode` above or below it keys exactly as it
would if the tee were absent; the tee never perturbs a cache key. The useful consequence:
on a **downstream cache hit**, the cached value is returned without pulling through the
tee, so the side-effect leg is never driven and the **write does not fire**. The write is
tied to the pull, and the cache short-circuits the pull. This is intended behavior, not a
gap.

### Write semantics: `create` and `append`

A sink writes in one of two ways, framed purely as write behavior:

- **`create`**: the target is (re)created from the written batches, replacing any prior
  contents.
- **`append`**: the written batches are added to the existing target.

This is the full surface: replace or add, nothing more.

### Durability and relationship to `into_backend`

A deferred write yields a **durable, user-named, user-owned** target. It is never
temporary and never auto-dropped; teardown is explicit. This is the concrete difference
from `RemoteTable`/`into_backend`, which produces an **anonymous, ephemeral** table dropped
in `clean_up` (`api.py:482`). They share the underlying write transport but are separate
constructs, and the API keeps them separate.

### Node shape

A sketch, not a contract; field names settle in Phase 1. Both nodes live in
`python/xorq/expr/relations.py` next to `Read`, `Tag`, and `CachedNode`.

```python
# python/xorq/expr/relations.py

class SinkNode(ops.Relation):
    parent: ops.Relation
    sink_method: str          # method invoked on the parent at execution time
    sink_kwargs: Any = ()     # mirrors Read.read_kwargs
    # terminal: produces no downstream relation

    def write(self):
        # fires the deferred write as a side effect; mirrors Read.make_dt
        return operator.methodcaller(self.sink_method, **dict(self.sink_kwargs))(
            self.parent.to_expr()
        )


class TeeNode(ops.Relation):
    parent: ops.Relation
    schema: Schema            # == parent.schema; passthrough
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
    sink_node = SinkNode(parent=parent, sink_method=..., sink_kwargs=(("mode", mode), ...))
    return TeeNode(parent=parent, schema=parent.schema, sink=sink_node).to_expr()
```

Two resolution passes keep transparency and the side effect separate:

- **Hashing**: a strip pass replaces each `TeeNode` with its parent before the hash is
  computed, like `_remove_non_hashing_tag_nodes` (`api.py:347`) for `Tag`.
- **Execution**: a transform pass wires the side-effect leg, mirroring
  `_register_and_transform_cache_tables` (`api.py:212`) and the RemoteTable fan-out
  (`relations.py:587`). It tees the parent batches with batchcorder (cardinality one), one
  replica downstream and the evicted replica through a `RecordBatchReader` into the
  `SinkNode`'s write.

### Naming

The user-facing function stays `.sink()` even though it builds a passthrough `TeeNode`,
not a terminal `SinkNode`. This bends the terminal connotation of "sink" on purpose, and
the documentation must say so up front: `.sink()` writes as a side effect and the pipeline
keeps going; it does not end the stream. The internal node names match their behavior:
`SinkNode` is the terminal write, `TeeNode` is the passthrough.

## Alternatives considered

### A single node that both writes and passes through

The earlier framing used one node that was simultaneously a terminal "sink" and a
passthrough tee.

**Rejected.** It conflates two behaviors with opposite shapes. "Sink" connotes terminal
(nothing flows past), while the common need is to write and continue. Splitting into a
terminal `SinkNode` and a passthrough `TeeNode` keeps each name honest and lets `.sink()`
compose them.

### Write unconditionally at execution, regardless of pull

Fire the write whenever a write node is present in the executed subgraph.

**Rejected.** It would fire on a pure cache hit, doing I/O for data nobody pulled. Tying
the write to the side-effect leg of the tee, which only runs when the main consumer pulls,
makes "cache hit means no write" fall out for free.

### Content-hash the tee on its write config

Fold the write configuration into the `TeeNode`'s token.

**Rejected.** It would change the cache keys of any cache above or below the tee, breaking
the transparency that lets sinks compose with caching at all. The node stays stripped
before hashing, like `Tag`.

### A dedicated N-way fan-out node

Provide one node that fans the stream out to N consumers directly.

**Deferred.** Phase 1 supports a fan-out of two (one main consumer, one side-effect
consumer); N is reached by chaining N-1 `TeeNode`s. A native N-way node can come later if
chaining proves insufficient.

## Consequences

### Positive

- A deferred write becomes the symmetric counterpart to `deferred_read_*`, with the same
  `method_name` plus kwargs shape.
- Splitting terminal `SinkNode` from passthrough `TeeNode` keeps the "sink is terminal"
  connotation intact while still letting a pipeline write and continue.
- TeeNode transparency means the write composes with caching with no special cases, and a
  downstream cache hit suppresses the write automatically.
- The two-node, fan-out-of-two design with chaining keeps Phase 1 small.

### Negative

- `.sink()` building a passthrough rather than a terminal node bends the "sink" name; this
  must be stated loudly in docs or users will expect `.sink()` to end the stream.
- Phase 1 leans on batchcorder buffering rather than true-tee blocking, so a side-effect
  consumer that does apply backpressure is not yet handled by blocking. Reaching the
  target tee behavior is follow-up work.
- A hard dependency on batchcorder (ADR-0013) sequences this work after it.

## References

- ADR-0013: batchcorder StreamCache for RemoteTable fan-out (forthcoming), hard dependency
- `Read` (deferred read precedent): `python/xorq/expr/relations.py:564`
- `deferred_read_*`: `python/xorq/common/utils/defer_utils.py`
- `Tag` / `HashingTag`: `python/xorq/expr/relations.py:101`
- Tag stripping pass: `_remove_non_hashing_tag_nodes`, `python/xorq/expr/api.py:347`
- RemoteTable fan-out (tee + `read_record_batches`): `register_and_transform_remote_tables`, `python/xorq/expr/relations.py:587`
- Cache resolution: `_register_and_transform_cache_tables`, `python/xorq/expr/api.py:212`
