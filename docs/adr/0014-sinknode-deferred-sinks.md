# ADR-0014: SinkNode, a transparent and durable lazy write tee

- **Status:** Proposed
- **Date:** 2026-06-09
- **Deciders:** Daniel
- **Context area:** `python/xorq/sinking/` (new), `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`, `python/xorq/vendor/ibis/expr/types/relations.py`

## Context

Xorq's caching system solves **memoization**: "have I computed this exact expression
before?" Each `execute()` of a given expression is keyed by a content hash; identical
expressions hit a cached file, different expressions miss and compute.

It does not solve **accumulation**: "append today's result to yesterday's." A daily
pipeline run produces a different expression every day (a new `snapshot_date` parameter,
new source data), so by construction each run is content-keyed to a different file. There
is no mechanism to write to a named, durable target across runs. dbt provides this with
`materialized='incremental'`.

This ADR proposes `SinkNode`, a new DAG node that defers a **write** decision to
execution time. It enables durable accumulation within Xorq's functional, immutable
expression model.

## Decision drivers

- A sink must **compose transparently** with the existing caching system: a cache
  upstream or downstream of a sink must continue to behave exactly as it would without
  the sink present.
- The SinkNode's write must respect the cache. If a downstream cache hits and the sink's
  data is never pulled, the sink must **not** write.

## Decision

### A SinkNode is transparent, like Tag

A `SinkNode` is a **transparent pass-through relation**, modeled on `Tag`
(`relations.py:101`). Its schema equals its parent's schema, and it evaluates to its
parent's rows unchanged. The only thing it adds is a **side effect**: as batches are
pulled through it during execution, they are written to a named, durable target.

Like `Tag`, a `SinkNode` is **stripped before hashing**: a dedicated pass replaces the
node with its parent before the content hash is computed, so `expr.sink(s)` and `expr`
produce the **same** hash. `_remove_non_hashing_tag_nodes` (`api.py:347`) does this for
`Tag`; `SinkNode` gets the parallel `_remove_sink_nodes` (see Node shape below). The
`"normalize_sink_node"` string in `__dasher_tokenize__` is only a tokenize label, not a
stripping function.

This transparency is the whole design. It is what makes the sink compose with caching
without special cases: because the node is invisible to the hash, a `CachedNode` above or
below a sink keys exactly as it would if the sink were not there. Nothing about the
cache's behavior changes when a sink sits next to it.

`CachedNode` and `SinkNode` differ in kind. A cache is content-keyed: it participates in
identity and can short-circuit a pull. A sink is transparent: it is invisible to identity
and only observes the pull.

| | hashing | evaluates to | side effect | lifetime |
| -- | -- | -- | -- | -- |
| `Tag` | stripped (transparent) | parent | none | n/a |
| `CachedNode` / `SourceStorage` | content hash | the cached/recomputed table | read-or-write cache | durable until `drop` |
| `RemoteTable` / `into_backend` | `gen_name()`, anonymous | relocated table | ephemeral table, dropped in `clean_up` | ephemeral |
| `SinkNode` / sink storage | **stripped (transparent, like `Tag`)** | **parent** | **write pulled batches to a user-named target** | **durable, never auto-dropped** |

### The SinkNode is a lazy tee: only pulled batches are written

A `SinkNode` is a **tee, not a terminal sink**: data flows through it to downstream
operations, and a copy of what flows is written to the target. `.sink()` resolves to its
parent's rows for this run, passed downstream transparently. 

The tee is **lazy**. It writes exactly the batches pulled through it during execution.
There is no unconditional, transform-time write; the pull-based execution that the
downstream operations request drives every write.

The direct consequence, and a deliberate requirement, is:

> **If a downstream operation hits a cache, the sink is not written.** A `CachedNode`
> downstream of a sink, on a cache hit, returns the stored value without descending into
> its parent. The sink's batches are therefore never pulled, so nothing is written.

A cache hit upstream still feeds the sink: the cached data is teed through as it is
pulled. A cache hit downstream leaves the sink's output undemanded, so the sink stays
silent.

| dbt concept | xorq equivalent |
| -- | -- |
| `materialized='table'` (full refresh) | `Sink(mode="create")` |
| `materialized='incremental'` (append) | `Sink(mode="append")` |

### Two modes: `create` and `append`

A sink has exactly **two modes**, set at construction as `Sink(mode=...)`:

- **`create`**: the target is (re)created from this run's pulled batches, replacing any
  prior contents. Maps to a full refresh, `CREATE OR REPLACE TABLE`.
- **`append`**: this run's pulled batches are added to the existing target. Maps to
  `INSERT`. On the first run the target is created.

`Sink.write(value)` writes `value` (the pulled batches) under the chosen mode. There is
no incremental loading: no `incremental_column`, no watermark filtering, no `unique_key`
anti-join, and no `merge` or dedup. The sink writes what it is given. Deduplication and
incremental windowing, when wanted, belong in the caller's expression upstream of the
sink.

This drops the RFC's incremental-strategy surface. The boundary rules, watermark
semantics, and re-pull-and-dedup machinery were the main source of the data-loss and
idempotency footguns. Idempotent reruns come from caching instead: a rerun hits the
cache, and the sink is left unwritten.

### Hashing

The `SinkNode` contributes nothing to the content hash. The sink's configuration (target
name or path, storage type, mode) is real metadata: build artifacts and catalog
serialization (Phase 3) record it so the write is reproducible. It lives **beside** the
content hash, not inside it, the same way a `Tag`'s metadata is recorded without changing
identity.

### Node shape

A sketch, not a contract; field names and helpers are settled in Phase 1. The node lives
in `python/xorq/expr/relations.py` next to `Tag` and `CachedNode`, and mirrors `Tag`: a
transparent relation carrying its parent, the parent's schema, and a side-effect config.

```python
# python/xorq/expr/relations.py

class SinkNode(ops.Relation):
    parent: ops.Relation
    schema: Schema            # == parent.schema; the node is a passthrough
    sink: Sink                # storage + target name + mode (config only)
    values = FrozenDict()     # transparent, like Tag

    @property
    def mode(self):
        return self.sink.mode

    def __dasher_tokenize__(self):
        # contributes nothing beyond the parent; stripped before hashing
        # (see _remove_sink_nodes), so expr.sink(s) hashes like expr
        return ("normalize_sink_node", self.parent)
```

```python
# python/xorq/sinking/ (new)

@frozen
class Sink:
    storage: SinkStorage      # ParquetSink | SourceSink | IcebergSink
    name: str                 # durable target name / path
    mode: str = "append"      # "create" | "append"

    @mode.validator
    def _check_mode(self, _, value):
        if value not in ("create", "append"):
            raise ValueError(f"mode must be 'create' or 'append', got {value!r}")

    def write(self, value):
        # value == the batches pulled through the node this run
        # create -> replace target; append -> add to target
        return self.storage.write(value, mode=self.mode)
```

```python
# python/xorq/expr/api.py : the user-facing attach

def sink(expr, storage, *, name, mode="append"):
    op = expr.op()
    cfg = Sink(storage=storage, name=name, mode=mode)
    return SinkNode(parent=op, schema=op.schema, sink=cfg).to_expr()
```

Two resolution passes keep the transparency and the side effect separate, mirroring the
cache machinery:

- **Hashing**: `_remove_sink_nodes` strips every `SinkNode` to its parent before the hash
  is computed, exactly as `_remove_non_hashing_tag_nodes` (`api.py:347`) does for `Tag`.
- **Execution**: `_register_and_transform_sink_nodes` wires the lazy-tee write of the
  pulled batches, mirroring `_register_and_transform_cache_tables` (`api.py:212`). The
  write fires only for batches the downstream operations actually pull.

### Composition with caching: allowed everywhere

Caching composes with sinks in **both directions**, with **no construction-time
restriction**:

- **Cache upstream** (`expr.cache(c).sink(s)`): the expensive compute is memoized first,
  and its result, whether freshly computed or served from cache, is pulled through the
  sink and written. This is the canonical arrangement.
- **Cache downstream** (`expr.sink(s).cache(c)`): allowed. On a cache miss the downstream
  cache pulls through the sink, the batches are written, and the result is cached. On a
  cache hit the cache returns the stored value without pulling, so the sink is not
  written. Accumulation does not advance on a pure cache hit, which is intended: the run
  that produced those rows already wrote them.

Cache hits and misses depend only on the parent content, since the sink does not change
the key. There is no stale-tee hazard, because the tee returns the parent's live rows
rather than a memoized delta. There is no write-suppression bug either: suppressing the
write on a cache hit is the correct outcome.

### Durability and relationship to `into_backend`

Sink targets are **durable by default, never temporary, never auto-dropped**; teardown is
only the explicit `SinkStorage.drop_all()`. A `SinkNode` does **not** enter the
`created`/`clean_up` drop path that `RemoteTable` uses (`api.py:482`). This durability is
the one concrete difference from `into_backend`: `into_backend`/`RemoteTable` is an
anonymous, ephemeral query-execution mechanic; a sink is a named, durable persistence
mechanic. They share the underlying write transport, but they are separate constructs and
the API keeps them separate.

### Naming

The node is `SinkNode`. The ADR records that "sink" carries a terminal connotation the
node deliberately violates (it is a transparent tee). Documentation must state this up
front. `AccumulateNode` was the runner-up; `SinkNode` was kept for its established RFC
usage, standard "durable write destination" vocabulary, and clean storage-subtype names
(`ParquetSink`, `SourceSink`, `IcebergSink`).

## Alternatives considered

### Content-hash the sink on its config

Fold `(parent_token, sink_config)` into the `SinkNode` token so the sink participates in
identity.

**Rejected** in favor of transparency. A config-hashed sink changes the cache keys of any
cache above or below it. That is what forced a construction-time ban on downstream
caching, to avoid the write-suppression and stale-tee hazards the key change created.
Modeling the sink on `Tag`, stripped before hashing, leaves the keys untouched, so the
hazards never arise and the ban is unnecessary. Transparency is both simpler and more
composable.

### Unconditional write at transform time

Fire the write whenever the `SinkNode` is present in the executed subgraph, regardless of
whether downstream operations pull its data.

**Rejected.** It re-fires the write on a pure cache hit, double-accumulating and wasting
I/O on rows the populating run already wrote. The lazy tee, which writes only what is
pulled, gives "cache hit means no write" for free.

### Incremental loading, `unique_key` dedup, and a `merge` strategy

Ship watermark filtering on an `incremental_column`, anti-join dedup on a `unique_key`,
and a `merge` strategy alongside `append`.

**Rejected** for Phase 1. The boundary rules (`>= max` versus `> max`), watermark advance,
and re-pull-and-dedup caused the data-loss and idempotency footguns, and they duplicate
logic the user can write upstream of the sink. Caching already handles idempotent
reruns. The mode surface stays at `create` and `append`.

### `.sink()` resolves to the full accumulated target

`.sink()` returns the whole target rather than passing the parent through.

**Rejected** because it breaks the transparent-tee model and composition: a transparent
node must evaluate to its parent. A separate API for reading the accumulated target back
is **deferred**, out of scope for now; until it lands, the target is read with an ordinary
`read_parquet` / table scan against its name.

### Mandate ADBC as the database write transport

**Deferred.** The write transport (ADBC versus `read_record_batches` versus native
`create_table`) is a per-`SinkStorage` implementation choice evaluated in Phase 2, not a
contract of the node semantics. The shared `write_to_source` primitive that would unify
the three existing write paths is out of scope. batchcorder's role (buffering
multi-consumer reads during a write) is an implementation detail, not a normative
decision.

## Consequences

### Positive

- Accumulation across runs becomes a first-class, immutable-expression-native operation.
- Transparency (stripped like `Tag`) makes sinks compose with caching in **both**
  directions: cache keys are computed as if the sink were absent, so there are no special
  cases and no forbidden combinations.
- The lazy tee makes "cache hit means no write" automatic, which removes
  double-accumulation along with the write-suppression and stale-tee hazards.
- Two modes (`create`, `append`) and no incremental machinery keep Phase 1 small and free
  of the watermark and dedup footguns.

### Negative

- `.sink()` returns the parent's rows, not the accumulated target. The "sink is a
  transparent tee, not terminal" point must be made loudly in docs, or users will expect
  `.sink()` to hand back everything written so far. A dedicated read-back API is deferred.
- Because the write only fires when data is pulled, a sink downstream of a cache that
  always hits will never advance accumulation. That is correct, but users must understand
  that a cache hit suppresses the write by design.
- No write-time dedup means `append` mode can accumulate duplicate rows across reruns
  unless caching short-circuits the rerun. Idempotency is the cache's job here, not the
  sink's.

## References

- RFC 0001: Deferred Sinks (Hussain, 2026-03-22)
- [XOR-252](https://linear.app/xorq-labs/issue/XOR-252): Deferred Sinks (epic)
- [XOR-262](https://linear.app/xorq-labs/issue/XOR-262): Phase 1, core primitives
- [XOR-263](https://linear.app/xorq-labs/issue/XOR-263): Phase 2, SourceSinkStorage, IcebergSinkStorage
- [XOR-264](https://linear.app/xorq-labs/issue/XOR-264): Phase 3, catalog serialization
- ADR-0013: batchcorder StreamCache for RemoteTable fan-out (forthcoming)
- `Tag` / `HashingTag`: `python/xorq/expr/relations.py:101`
- `CachedNode` / `RemoteTable`: `python/xorq/expr/relations.py`
- Tag stripping pass: `_remove_non_hashing_tag_nodes`, `python/xorq/expr/api.py:347`
- Cache resolution: `_register_and_transform_cache_tables`, `python/xorq/expr/api.py:212`
- dbt incremental materialization docs
