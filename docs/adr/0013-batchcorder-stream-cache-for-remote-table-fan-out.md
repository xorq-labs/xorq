# ADR-0013: Replace SafeTee with batchcorder StreamCache for RemoteTable fan-out

- **Status:** Accepted
- **Date:** 2026-05-22
- **Deciders:** Daniel
- **Context area:** `python/xorq/expr/remote_table_exec.py`, `python/xorq/expr/api.py`, `python/xorq/expr/relations.py`

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

`register_and_transform_remote_tables` (`remote_table_exec.py`) walks an expression graph,
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

- **Correctness depended on an exact pre-count.** `SafeTee` allocated exactly `n`
  tee copies up front, so an inaccurate `n` was a *correctness* bug, not a tuning
  miss: undercounting exhausted the tee pool (empty results), overcounting leaked
  unconsumed iterators. But the scan count is not part of DuckDB's public API — it
  varies by query shape, DuckDB version, and optimiser decisions (ASOF joins,
  self-joins, and joins with intermediate aggregations all re-scan). A count that
  cannot be known exactly must not gate correctness. (We still pre-count under
  batchcorder, but only as an eviction hint — see `max_readers` below.)
- **Thread safety is incomplete.** `itertools.tee` is explicitly documented as not
  thread-safe. DuckDB scans registered tables from parallel threads within a single
  query.
- **`created` cardinality was inflated.** The function returned one entry per tee
  copy, so a self-join with `n=3` copies produced `len(created) == 3` even though
  only one remote table was involved.

## Decision drivers

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

In `replacer` (inside `register_and_transform_remote_tables` in
`remote_table_exec.py`), each `RemoteTable` gets a single `StreamCache` whose
resources are tracked by a `RemoteTableScope`:

```python
reader = scope.adopt_reader(remote_expr.to_pyarrow_batches())
cache = scope.adopt_cache(
    StreamCache(reader, max_readers=reader_counts.get(node))
)
table_name = scope.adopt_table(node.source, gen_name())
result = node.source.read_record_batches(cache, table_name=table_name, **read_kwargs)
```

DuckDB acquires a fresh `StreamCacheReader` via `__arrow_c_stream__` each time it
scans the registered table. Every reader starts at batch 0 and advances
independently. The upstream remote expression is called exactly once — ingestion is
lazy and shared across all readers via an internal `Arc<Mutex<DatasetInner>>`.

This is what makes `SafeTee` unnecessary. Because the registered table is replayable,
a self-join over a `RemoteTable` works without any caller-side tee allocation: each
join side acquires its own `StreamCacheReader` and replays the same stream from
batch 0. The same holds for ASOF-with-`tolerance` and any other multi-scan shape.

#### Bounded eviction via `max_readers`

`count_remote_table_readers` compiles the expression to a sqlglot AST with sentinel
table names and counts how many `Table` nodes bear each sentinel. The count becomes
`StreamCache`'s `max_readers`, allowing it to evict batches once all readers advance
past them. When no AST can be produced (non-SQL backend), `max_readers` is omitted and
the cache retains all batches.

This count is a best-effort *lower bound*, not an exact physical scan count. It sees
only the scans the compiled SQL spells out; it cannot see re-scans the backend's
optimiser introduces *below* the SQL layer. The known gap: DuckDB lowers a
`PARTITION BY`-only aggregate window (`sum(v) OVER (PARTITION BY k)`, no `ORDER BY`)
into a `GROUP BY` self-join — `HASH_JOIN(SEQ_SCAN, GROUP_BY(SEQ_SCAN))`, no `WINDOW`
operator — that scans its input twice, while the AST counts one `Table` node.

`max_readers` is enforced as a **hard cap**: a reader beyond the cap raises
`ValueError: Maximum number of readers reached`. So an *undercount* is not merely a
missed memory optimisation — it is a correctness bug for the query shapes it misses
(the partition-window case above is one). Omitting the count entirely is always safe
(unbounded cache); deriving a count that is too low is not. This is the one place the
batchcorder design still depends on getting a scan count right, and the AST heuristic
does not get it right for every shape. (See "Counting accuracy" under Negative.)

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

### Resource lifecycle — `RemoteTableScope`

`register_and_transform_remote_tables` returns `(expr, scope)` where
`scope: RemoteTableScope` owns every resource the transform materialised:
upstream `RecordBatchReader`s, `StreamCache`s, and placeholder tables.
`scope.close()` tears down in dependency order (tables → caches → readers),
LIFO within each category, and is idempotent.

Call sites use one of two patterns depending on whether execution is eager or
streaming:

- **Eager (fully materialised before returning):**
  `remote_table_scope` is a context manager in `api.py` that yields the
  transformed expression and closes the scope on exit.  Used by
  `_pandas_execute`, `get_plans`, and any backend whose
  `read_record_batches` eagerly ingests the stream (Snowflake, Postgres via
  ADBC through `prepare_create_table_from_expr`).

- **Streaming (`to_pyarrow_batches`):**
  `bind_scope_to_reader` wraps the result `RecordBatchReader` in a generator
  whose `finally` block calls `scope.close()`.  A `weakref.finalize`
  backstop covers readers abandoned before the first read (a never-started
  generator never runs its `finally`).

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

- **Replay is decoupled from the count.** `replacer` creates one `StreamCache` per
  `RemoteTable` node, and any reader can replay the full stream from batch 0. The
  sqlglot-AST count feeds only `max_readers`, which bounds eviction; omitting it is
  always safe — the cache simply retains all batches. (The count is *not* free of
  correctness consequences when supplied — see "Counting accuracy" below.)
- **Thread-safe reads.** Concurrent `StreamCacheReader` handles share an
  `Arc<Mutex<DatasetInner>>` and are designed for concurrent use.
- **Prompt resource release.** `RemoteTableScope.close()` releases tables, caches,
  and readers in dependency order. Eager call sites use `remote_table_scope`;
  streaming call sites use `bind_scope_to_reader`.
- **Fixes #983 and most of the multi-scan class.** ASOF join with `tolerance` and
  self-join both work correctly. The replay mechanism itself handles any number of
  scans; the remaining gap is purely in `max_readers` undercounting certain shapes
  (see "Counting accuracy").

### Negative

- **New runtime dependency.** `batchcorder == 0.1.3` is now required. It ships
  pre-built wheels, requires only `pyarrow` (no `arro3`), and supports Python ≥ 3.10.
- **Counting accuracy (`max_readers`).** The sqlglot-AST scan count is a heuristic
  lower bound, and `max_readers` is a hard cap, so any query whose backend re-scans a
  source below the SQL layer can undercount and raise `ValueError: Maximum number of
  readers reached`. The known case is a `PARTITION BY`-only aggregate window on DuckDB
  (lowered to a `GROUP BY` self-join). Reading the backend's physical `EXPLAIN` plan
  would give the exact count, but that plan is data-dependent (the optimiser prunes
  scans of provably-empty placeholder tables), so it cannot be reproduced offline
  without representative data; pending a fix, the AST count stands.
- **Abandoned reader leak (streaming path).** `bind_scope_to_reader` defers cleanup
  to a wrapping generator's `finally` block, with a `weakref.finalize` backstop.
  Python guarantees the finaliser runs when the reader is GCed, but the timing is
  non-deterministic for callers that drop the reader early. In memory-only mode this
  is acceptable; disk mode would require deterministic cleanup (e.g. wrapping the
  reader in a context manager).

## References

- [Issue #983](https://github.com/xorq-labs/xorq/issues/983) — `asof_join` with `tolerance` and `into_backend` gives empty result
- `python/xorq/expr/remote_table_exec.py` — `RemoteTableScope`, `bind_scope_to_reader`, `register_and_transform_remote_tables`, `count_remote_table_readers`, `prepare_create_table_from_expr`
- `python/xorq/expr/api.py` — `_transform_expr`, `remote_table_scope`, `to_pyarrow_batches`, `_pandas_execute`, `get_plans`
- `python/xorq/expr/relations.py` — `RemoteTable`
- [batchcorder on PyPI](https://pypi.org/project/batchcorder/)
