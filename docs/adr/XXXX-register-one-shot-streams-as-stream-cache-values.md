# ADR-XXXX: Register one-shot Arrow streams as StreamCache values and key them by content

- **Status:** Proposed
- **Date:** 2026-11-26
- **Deciders:** Daniel
- **Related:** [ADR-0013](0013-batchcorder-stream-cache-for-remote-table-fan-out.md), [ADR-0015](0015-build-hash-cache-hash-split.md), [ADR-0016](0016-table-driven-opaque-descent-with-registration-tripwires.md), [ADR-0017](0017-canonical-hash-forms-not-serializer-bytes.md)

## Context

[Issue #2674](https://github.com/xorq-labs/xorq/issues/2674): a table registered from a `pa.RecordBatchReader` on DataFusion gets a cache key built from its schema and name only (`python/xorq/common/utils/dasher/_relations.py:254`). Two readers with different rows under the same name share a key, and `.cache()` serves the first reader's rows. No error is raised.

Probing found a second bug with the same cause on DuckDB ([#2719](https://github.com/xorq-labs/xorq/issues/2719)). There, `ARROW_SCAN` already routes to `normalize_memory_databasetable_canonical` (`python/xorq/common/utils/dasher/_canonical.py:346`), which runs the table to hash its rows. Against a one-shot reader, computing the key reads the stream to the end:

```
uncached:              [2, 4, 6]
tokenize then execute: []        # key computed, data gone
cached:                []
```

What the two have in common: **a one-shot iterator is not a value.** The rows can't be read without consuming them, so any key derived from the rows uses up the data the key describes. Every option below handles this effect somewhere different.

The constraints:

- The key has to exist **before** execution. `Cache.set_default` (`python/xorq/caching/__init__.py:109`) computes it to decide hit or miss. A digest built while the consumer streams finishes only at the end of the stream, which is too late for the lookup.
- The plan doesn't show whether a source is one-shot. When registered with `read_record_batches`, a `pa.Table`, a `RecordBatchReader` and a `StreamCache` all plan as `StreamingTableExec` on DataFusion and `ARROW_SCAN` on DuckDB, yet only the reader is one-shot:

  | source | DuckDB | DataFusion `read_record_batches` |
  |---|---|---|
  | `pa.Table` | replayable | one-shot ([#2720](https://github.com/xorq-labs/xorq/issues/2720)) |
  | `RecordBatchReader` | one-shot | one-shot |
  | `StreamCache` | replayable | replayable |

- Memory-backed tables already use a canonical, batch-boundary-free digest (ADR-0017, #2191). Folding a hash over the batch stream would bring back the chunking dependence that digest removed.

## Decision drivers

- Correct keys: different rows give different keys, and the same rows give the same key whatever the batch chunking.
- Computing a key must never change what a later execution returns.
- No mutable state on the backend or in a module-level registry to track which tables are streams.
- Lazy: a table that is never cached does no hashing work.
- Replayable sources keep their current behaviour.

## Decision

### Make the stream a value at the boundary

Every path that would register a one-shot stream registers `StreamCache(stream)` instead, in memory mode with no `max_readers`:

- DataFusion `register`: a `pa.RecordBatchReader`, and the reader produced for a cross-backend non-table expression.
- DataFusion `read_record_batches`: the casting reader it builds for a `pa.Table`, a `pa.RecordBatchReader` or an iterable of batches. The `pa.Table` case is included because that reader is one-shot too (#2720), and keying it by content would otherwise use it up.
- DuckDB `read_record_batches`: a `pa.RecordBatchReader`. DuckDB already reads a `pa.Table` more than once.

Sources that are already replayable are not wrapped: `StreamCache`, `CastingStreamCache` and `TeeTableProvider` go through unchanged, so the remote-table pass (ADR-0013) doesn't buffer its data twice.

`StreamCache` reads its upstream lazily. Registering pulls nothing. The upstream gets read once, by whichever reader pulls first (hashing or execution), and every later reader replays from batch 0.

### Key it like any other memory-backed table

The DataFusion `StreamingTableExec` branch in `_normalize_datafusion_databasetable_xorq` returns `normalize_memory_databasetable_canonical(dt)`. That is the rule DuckDB `ARROW_SCAN`, pandas and in-memory SQLite tables already use. The name-only tuple `("ibis.DatabaseTable.datafusion.recordbatch", schema, name)` is removed. DuckDB needs no dasher change: wrapping at the boundary is what makes its existing rule safe.

Because the registered source is a value, hashing is a plain read: no effect on the backend, no rebinding, and it works through `con.table(name)` as well as through the expression `register` returned. All of this was checked on both backends against a counting upstream:

```
register  -> 0 upstream pulls
hash      -> 1 pull; hashing again via con.table("rb") gives the same key
execute x2 -> [2, 4, 6] [2, 4, 6], 1 upstream pull total
different rows -> different key
```

### Retention

The number of future readers of a user-registered table can't be known, so the cache keeps every batch for the table's lifetime. Memory mode shares batches by `Arc`, so N replays hold one copy, not N. The backend holds the only reference to the cache, so deregistering the table or registering the name again frees it.

That is a cost compared with today. A raw reader streams in O(batch), while a wrapped one keeps the whole input. The cached path already pays for it, because the canonical digest needs every column resident. The extra cost lands on uncached, single-execution reads of large streams.

The planned bound is disk mode with `write_policy="on_eviction"` and a `memory_capacity` budget, which batchcorder supports today. It is not the default yet because xorq would have to own the spill directory's lifecycle. [mesejo/batchcorder#55](https://github.com/mesejo/batchcorder/issues/55) asks batchcorder to own that directory, which makes the switch a configuration change.

### Behaviour change

A table registered from a one-shot reader can now be executed more than once. Today, a second execution silently returns `[]`. This is intended: a silent empty result was never a contract anyone depended on.

## Alternatives considered

### Key on schema and name (status quo)

Rejected because:
- It is the bug.

### Refuse to cache stream-backed tables

`.cache()` over a reader-backed table raises and points the user at `pa.Table` or a file.

Rejected because:
- A stream whose rows are the same should hit the cache, and refusing makes caching useless for this source.
- It doesn't fix the DuckDB key computation that empties the stream, which happens with or without `.cache()` whenever a key is computed.

### Per-registration identity (uuid)

Mint a uuid per `register(reader)` call and fold it into the key.

Rejected because:
- It is safe but never hits across registrations, so in practice it is the same as refusing to cache.
- It brings back the non-reproducible key class fixed in gh-2229.

### Caller-supplied key: `register(reader, cache_key=...)`

Rejected because:
- It puts correctness on the caller, and a wrong token gives back this bug.
- Once content keying exists, it adds API surface with no clear need.

### Incremental hash while the consumer streams

Fold each batch into a hasher as execution reads it.

Rejected because:
- The digest is ready only at end of stream, but the key is needed before execution to decide hit or miss.
- A per-batch fold depends on chunking (same rows in different batches give different keys). `_canonical.py:346` already rejects this for memory tables, and ADR-0017 is the rule behind that.

### Tee at hash time and rebind the name

When the key is computed, `ds = StreamCache(con.table(name).to_pyarrow_batches(), max_readers=2)`, hash `ds.reader()`, then re-register `ds.reader()` under the same name. Exactly two readers, so the cache evicts once the table reader has read the stream and nothing is retained. Checked on both backends: it works, and computing the key twice gives the same result.

Rejected because:
- **Computing the key has a side effect.** It consumes the registration and rebinds it. A pure version would need `tokenize: (expr, con) -> (key, con')`, and dasher threads no state.
- **It needs to know the source is one-shot.** The plan can't tell (table above). Running the tee on a replayable `pa.Table` or `StreamCache` turns it into a one-shot source. Knowing requires state on the backend or a node subclass, and the subclass still leaves `con.table(name)` uncovered.
- **The key depends on order.** Computing it after an execution hashes the drained, empty stream, so the key reflects when you asked rather than which rows were registered.
- **Race.** Two concurrent key computations both consume and rebind the same name.
- Registering the `StreamCache` itself instead of `ds.reader()` avoids the downgrade, but every later key computation then wraps another cache around the first.

### Mark stream tables with a node subclass (`StreamTable(ops.DatabaseTable)`)

Following the `PyIcebergTable` precedent, `register(reader)` returns a subclass, and a `ViewRule` row dispatches it to the tee-at-hash-time rule.

Rejected because:
- It only exists to support the rejected tee-at-hash-time rule. Once the source is a value, plain `DatabaseTable` dispatch is enough.
- `con.table(name)` returns a plain `DatabaseTable` and skips the rule, so the silent wrong-key path stays open.

### Track one-shot tables on the backend (`{name}` set or `{name: (cache, digest)}`)

Rejected because:
- It is mutable state outside the expression, which has to be invalidated by hand on every re-registration. A stale digest would bring this bug back.

### Wrap with a disk-backed cache by default

Deferred because:
- xorq would have to own a temp directory's lifecycle and pay IPC writes on every stream table. Revisit when [mesejo/batchcorder#55](https://github.com/mesejo/batchcorder/issues/55) lands.

### Late-bound reader cap

Deferred because:
- It doesn't help user-registered tables, since their future readers are never known. It does help xorq-owned temporary caches: with `seal()` after the engine has planned and opened its scans, the remote-table pass could drop the SQL-based reader count and its undercount failure (ADR-0013, "Counting accuracy"). Filed as [mesejo/batchcorder#56](https://github.com/mesejo/batchcorder/issues/56).

### Other batchcorder levers

Considered, not filed:
- Compressing retained batches in memory (IPC lz4/zstd): 2–5x smaller retention, at a CPU cost on every replay.
- Computing the canonical digest during ingestion in Rust: a boundary-free fold over logical column values would remove the hash's own full materialization and `combine_chunks` copy. It doesn't change retention. Dictionary and nested types need canonicalization first, which makes this the largest piece of work in the list.

## Consequences

### Positive

- Stream-backed tables are keyed by content on DataFusion and DuckDB, and the issue's repro gets different keys.
- On DuckDB, computing a key no longer empties a one-shot reader.
- No new op, no backend registry, no new dasher rule: one branch now points at the existing memory-canonical rule.
- `con.table(name)` and the expression returned by `register` give the same key.
- One-shot tables can be executed more than once.

### Negative

- The whole input is retained in RAM for the table's lifetime, and uncached single-execution reads of large streams take the memory hit. Bounded later by disk mode (#55).
- Hashing executes the table once more and holds it fully resident (the canonical digest's existing cost, now applied to streams too).
- A visible behaviour change: a second execution returns rows where it used to return `[]`.
- DuckDB's deprecated vendored `read_in_memory(reader)` still registers the raw reader and keeps its own consumed-reader warning. It is left as it is because it is slated for removal.

## References

- [Issue #2674](https://github.com/xorq-labs/xorq/issues/2674): tables registered from a RecordBatchReader are cached by schema and name only
- [Issue #2719](https://github.com/xorq-labs/xorq/issues/2719): DuckDB key computation consumes a RecordBatchReader
- [Issue #2720](https://github.com/xorq-labs/xorq/issues/2720): DataFusion `read_record_batches(pa.Table)` can be executed only once
- [Issue #2191](https://github.com/xorq-labs/xorq/issues/2191): pyarrow-version-coupled memory-table hash
- [mesejo/batchcorder#55](https://github.com/mesejo/batchcorder/issues/55): disk mode owns a temporary directory
- [mesejo/batchcorder#56](https://github.com/mesejo/batchcorder/issues/56): late-bound reader cap
- `python/xorq/common/utils/dasher/_relations.py`: `_normalize_datafusion_databasetable_xorq`, `_normalize_duckdb_databasetable_xorq`
- `python/xorq/common/utils/dasher/_canonical.py`: `normalize_memory_databasetable_canonical`
- `python/xorq/backends/xorq_datafusion/backend.py`: `register`, `read_record_batches`
- `python/xorq/backends/duckdb/__init__.py`: `read_record_batches`
