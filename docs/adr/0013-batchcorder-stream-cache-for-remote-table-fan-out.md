# ADR-0013: Replace SafeTee with batchcorder StreamCache for RemoteTable fan-out

- **Status:** Accepted
- **Date:** 2026-05-22
- **Deciders:** Daniel
- **Context area:** `python/xorq/expr/relations.py`, `python/xorq/expr/api.py`, `python/xorq/backends/pandas/__init__.py`

## Context

### Motivating bug

[Issue #983](https://github.com/xorq-labs/xorq/issues/983): `asof_join` with `tolerance` and `into_backend` returns an empty result.

```python
con = xo.duckdb.connect()
sensors_bt = sensors.into_backend(con)
events_bt  = events.into_backend(con)

expr = (
    sensors_bt
    .asof_join(events_bt, on="event_time", predicates="site", tolerance=timedelta(seconds=1))
    .drop("event_time_right")
    .order_by("event_time")
)
expr.execute()  # empty — should have 5 rows
```

### Root cause: multi-scan of a one-shot Arrow iterator

`register_and_transform_remote_tables` (`relations.py`) walks an expression graph,
finds every `RemoteTable` node, materialises the remote expression as an Arrow
`RecordBatchReader`, and registers it with the local DuckDB backend so the expression
can execute locally.

The upstream reader is a one-shot iterator. DuckDB's ASOF join with `tolerance`
internally scans the registered Arrow source more than once. The first scan exhausts
the iterator; subsequent scans find nothing and produce an empty result.

This is a class of bug, not a single query shape. Any DuckDB operation that re-scans
a registered Arrow source hits it. Two confirmed cases:

1. **ASOF join with `tolerance`** — DuckDB re-scans to apply the tolerance filter
   (issue #983).
2. **Self-join** — both sides of the join point to the same `RemoteTable`; each side
   tries to advance the same iterator.

### Previous solution: SafeTee

```python
class SafeTee:
    def __init__(self, iterable, n):
        with threading.Lock():
            self._tees = itertools.tee(iterable, n)
    def __iter__(self):
        return iter(self._tees.pop())
```

`SafeTee` required the caller to know `n` — the total number of times the same
`RemoteTable` would be consumed — before allocating the tee copies. This forced a
two-pass graph walk:

1. Walk the full graph, counting occurrences of each `RemoteTable` expression hash.
2. Walk again, allocating `SafeTee(upstream, count)` on first encounter and popping
   one tee copy per subsequent encounter.

Problems:

- **Pre-counting is fragile.** DuckDB can scan a registered source more than once
  inside a single query (ASOF joins, self-joins, joins with intermediate aggregations).
  The scan count is not part of DuckDB's public API and varies by query shape,
  DuckDB version, and optimiser decisions. Undercounting exhausted the tee pool;
  overcounting leaked unconsumed iterators.
- **Thread safety is incomplete.** `itertools.tee` is explicitly documented as not
  thread-safe. DuckDB scans registered tables from parallel threads within a single
  query.
- **`created` cardinality was inflated.** The function returned one entry per tee
  copy, so a self-join with `n=3` copies produced `len(created) == 3` even though
  only one remote table was involved.

## Decision drivers

- Multi-scan must be transparent — no pre-counting at call sites.
- Concurrent reads (DuckDB scanning from multiple threads) must be safe.
- Memory and disk resources must be released promptly after execution.

## Decision

### Replace SafeTee with batchcorder StreamCache

[batchcorder](https://pypi.org/project/batchcorder/) is a Rust-backed library
authored by Daniel Mesejo specifically to solve this class of problem. Its
`StreamCache` wraps any Arrow stream source in a replayable cache: it ingests
batches lazily from the upstream reader and stores them internally. Any number of
independent `StreamCacheReader` handles can replay the full stream from position zero
without knowing the total reader count in advance.

In `replacer`, each `RemoteTable` now gets a single `StreamCache`:

```python
cache = StreamCache(
    pa.RecordBatchReader.from_batches(schema, remote_expr.to_pyarrow_batches())
)
caches.append(cache)
table_name = gen_name()
result = node.source.read_record_batches(cache, table_name=table_name)
created[table_name] = node.source
```

DuckDB acquires a fresh `StreamCacheReader` via `__arrow_c_stream__` each time it
scans the registered table. Every reader starts at batch 0 and advances
independently. The upstream remote expression is called exactly once — ingestion is
lazy and shared across all readers via an internal `Arc<Mutex<DatasetInner>>`.

### Storage modes

`StreamCache` supports two modes:

- **Memory-only** (default): batches stored as `Arc<RecordBatch>` in a Rust `Vec`.
  Reads are zero-copy Arc clones. xorq uses this mode today.
- **Disk** (`disk_path` + `disk_capacity` both provided): batches serialised to an
  append-only Arrow IPC file with a configurable hot layer in RAM.

Disk mode is not currently used but is planned as a future configuration option for
streams that exceed available RAM. Because `close()` is already called explicitly at
all executing call sites, enabling disk mode later is a configuration change only —
no new cleanup logic is required.

### Resource lifecycle

`register_and_transform_remote_tables` returns `(expr, created, caches)` where
`caches: list[StreamCache]` is owned by the call site.

**Executing call sites close caches explicitly:**

- **`_pandas_execute`**: `finally` block immediately after `con.execute()` returns.
  Pandas execution is synchronous and fully materialises before returning.
- **`get_plans`**: `finally` block after the `EXPLAIN` SQL completes. The upstream
  stream is never read during `EXPLAIN`; closing here releases it without waiting
  for GC.
- **`to_pyarrow_batches`**: `cache.close()` inside the `clean_up` callback passed
  to `rbr_wrapper`. The callback fires when the returned `RecordBatchReader` generator
  exits — either on full exhaustion or when the generator is closed/GCed.
- **`prepare_create_table_from_expr`**: `finally` block after the transformed
  expression is returned. Caches are closed before the caller materialises the table.

### Effect on `created`

`created` maps `table_name → backend`. For a self-join that previously generated
three `SafeTee` copies, `len(created)` drops from 3 to 1 — one entry per unique
`RemoteTable` node.

## Alternatives considered

### Keep SafeTee with accurate pre-counting

Audit every DuckDB code path to determine the exact scan count per query shape and
increase it to account for internal re-scans.

Rejected: DuckDB's scan count is not part of its public API and varies by query
shape, DuckDB version, and optimiser. Any hardcoded count breaks silently on upgrade.
Even with a correct count, concurrent scans require per-copy locking that `SafeTee`
did not provide.

### `itertools.tee` directly

Drop the `SafeTee` class and call `tee` directly.

Rejected: `itertools.tee` is explicitly documented as not thread-safe. DuckDB scans
registered tables from parallel threads within a single query.

### Pre-ingest into a `pa.Table`

Call `remote_expr.to_pyarrow()` to materialise the full result, register the table
once, and let DuckDB read from it N times.

Rejected: forces full materialisation before any execution begins, eliminating lazy
streaming. For large remote results the full table must fit in memory before DuckDB
can begin processing. `StreamCache` provides the same multi-read behaviour with lazy
ingestion and an optional disk spill path.

## Consequences

### Positive

- **No pre-counting.** `replacer` creates one `StreamCache` per `RemoteTable` node
  without knowing how many times DuckDB will scan it.
- **Thread-safe reads.** Concurrent `StreamCacheReader` handles share an
  `Arc<Mutex<DatasetInner>>` and are designed for concurrent use.
- **Prompt resource release.** Explicit `close()` at execution boundaries releases
  memory immediately rather than waiting for GC.
- **`created` cardinality is natural.** One entry per unique `RemoteTable` node.
- **Fixes #983 and the full multi-scan class.** ASOF join with `tolerance` and
  self-join both work correctly.

### Negative

- **New runtime dependency.** `batchcorder >= 0.1.2` is now required. It ships
  pre-built wheels, requires only `pyarrow` (no `arro3`), and supports Python ≥ 3.10.
- **Abandoned reader leak (generator close path).** `clean_up` in `to_pyarrow_batches`
  fires inside the generator's `finally` block. Python guarantees this runs when the
  generator is closed or GCed, but the timing is non-deterministic for callers that
  drop the reader early. In memory-only mode this is acceptable; disk mode would
  require deterministic cleanup (e.g. wrapping the reader in a context manager).

## References

- [Issue #983](https://github.com/xorq-labs/xorq/issues/983) — `asof_join` with `tolerance` and `into_backend` gives empty result
- `python/xorq/expr/relations.py` — `register_and_transform_remote_tables`, `replacer`, `prepare_create_table_from_expr`
- `python/xorq/expr/api.py` — `_transform_expr`, `to_pyarrow_batches`, `_pandas_execute`, `get_plans`
- `python/xorq/backends/pandas/__init__.py` — `read_record_batches` StreamCache branch
- `python/xorq/common/utils/defer_utils.py` — `rbr_wrapper`
- [batchcorder on PyPI](https://pypi.org/project/batchcorder/)
