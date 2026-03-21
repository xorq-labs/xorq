# ADR-0003: Use isolated DuckDB cursors for `into_backend` source reads

- **Status:** Accepted
- **Date:** 2026-03-20
- **Context area:** `python/xorq/backends/duckdb/__init__.py`, `python/xorq/expr/relations.py`

## Context

`into_backend` transfers data between backends by wrapping source expressions in `RemoteTable` nodes. At execution time, `register_and_transform_remote_tables` materializes these nodes: it calls `to_pyarrow_batches()` on each source expression to get a streaming `RecordBatchReader`, then registers that reader on the target backend.

The function uses a two-phase approach:

1. **Open phase** — iterate over all `RemoteTable` nodes, calling `to_pyarrow_batches()` on each source expression to create streaming readers
2. **Register phase** — during expression graph replacement, wrap each reader in a `RecordBatchReader` and register it on the target via `read_record_batches()`

DuckDB's `to_pyarrow_batches` calls `self.con.sql(sql).fetch_arrow_reader(chunk_size)`, which opens a **live streaming result** on the connection handle. DuckDB only supports one active streaming result per connection handle. When the open phase processes the second `RemoteTable` from the same source connection, the new `con.sql()` call silently invalidates the first reader, causing it to yield zero batches.

This produced three bugs:

| Source | Target | Same con? | Result |
|--------|--------|-----------|--------|
| DuckDB | DuckDB | yes | Deadlock — source read and target registration both need the same handle |
| DuckDB | DuckDB | no | Empty results — second cursor invalidates the first, no error raised |
| DuckDB | DataFusion | N/A | Empty results — same invalidation, most dangerous because cross-backend federation is the primary `into_backend` use case |

DataFusion was unaffected as the source because its `to_pyarrow_batches` returns a lazy generator backed by a plan object, not a live cursor. Multiple readers coexist without invalidation.

### Constraints

Several alternative approaches were evaluated and rejected:

- **Eager materialization** (`list(to_pyarrow_batches())`) — consumes the cursor immediately but loads the entire dataset into Python heap memory. For large tables this causes OOM, since Python has no spill-to-disk mechanism.

- **Global cursor in `to_pyarrow_batches`** (always use `con.cursor()`) — DuckDB cursors cannot see replacement scans created by `con.register()` or temporary tables. This breaks normal query execution, where `execute()` calls `to_pyarrow_batches` on expressions that reference registered in-memory tables.

- **Sequential processing** (open one cursor, register, consume, then open the next) — requires the target's `read_record_batches` to eagerly consume the reader before the next cursor opens. DuckDB's `con.register()` is lazy (data is consumed at query time, not registration time), and DataFusion's `register_record_batch_reader` is also lazy. Forcing eager consumption would require creating DuckDB tables (`CREATE TABLE ... AS SELECT`), which materializes data, and modifying every target backend.

- **Locks** — the problem is not concurrent access (everything is single-threaded) but resource lifecycle: multiple streaming results are *created* before any are *consumed*. A lock cannot change this ordering without restructuring the loop, at which point it reduces to sequential processing.

## Decision

Add an `isolated` parameter to `to_pyarrow_batches` on the DuckDB backend. When `isolated=True`, the method uses `self.con.cursor()` to obtain a dedicated DuckDB cursor for the query, so that multiple readers from the same connection coexist without invalidation.

`register_and_transform_remote_tables` calls the source backend's `to_pyarrow_batches` directly (bypassing the `Expr` dispatch) with `isolated=True`, but only when the source backend is DuckDB (`source_backend.name == "duckdb"`). Non-DuckDB backends never receive the parameter.

### Changes

1. **`duckdb/Backend.to_pyarrow_batches`** — new `isolated: bool = False` parameter. When true, calls `self.con.cursor().sql(sql).fetch_arrow_reader(chunk_size)` instead of `self._to_duckdb_relation(expr).fetch_arrow_reader(chunk_size)`. The default path (`isolated=False`) is unchanged. The existing `**_: Any` absorbs the parameter when passed to the base class.

2. **`register_and_transform_remote_tables`** — resolves the source backend via `ex._find_backend()` and calls `source_backend.to_pyarrow_batches(ex, isolated=True)` only when `source_backend.name == "duckdb"`. For all other backends, calls `source_backend.to_pyarrow_batches(ex)` without the parameter. This avoids leaking a DuckDB-specific kwarg to backends that may forward `**kwargs` to internal methods (e.g., DataFusion forwards kwargs to `compile()`).

## Rationale

### Why a parameter, not a separate method?

An earlier iteration added `to_pyarrow_batches_isolated()` as a separate method, with `hasattr`-based dispatch in `register_and_transform_remote_tables`. This was rejected because:

- Protocol-based `hasattr` checks couple generic graph-rewriting code to backend implementation details
- Every backend with similar constraints would need to add the same ad-hoc method
- A parameter on the existing method is discoverable, type-checkable, and follows the existing pattern of keyword arguments that backends selectively handle

### Why not change `to_pyarrow_batches` globally?

DuckDB cursors (`con.cursor()`) are independent connection handles that share the database catalog (regular tables) but **not** connection-scoped state:

- Replacement scans (from `con.register()`) are invisible to cursors
- Temporary tables are invisible to cursors

The vendored Ibis DuckDB backend uses `con.register()` in `read_in_memory`, `_register_in_memory_table`, and `read_record_batches`. If `to_pyarrow_batches` always used a cursor, queries referencing these registered objects would fail with `CatalogException: Table does not exist`.

The `isolated` flag restricts cursor usage to the one call site that needs it — source reads during `RemoteTable` materialization — where the compiled SQL references only the source backend's own tables (not registered objects on the target).

### Why call the source backend directly?

`ex.to_pyarrow_batches()` dispatches through `api.to_pyarrow_batches`, which calls `_transform_expr` (and therefore `register_and_transform_remote_tables` recursively). While the recursion is a no-op for source expressions without `RemoteTable` nodes, it adds unnecessary overhead. Calling `source_backend.to_pyarrow_batches(ex, isolated=True)` directly is both more efficient and more explicit.

## Consequences

### Positive

- All three DuckDB `into_backend` bugs are fixed (same-con deadlock, different-con empty results, cross-backend empty results) without materializing data or changing the streaming execution model.
- No changes to `register_and_transform_remote_tables` graph replacement logic — only the batch-fetching loop is modified.
- The `isolated` parameter is opt-in and backward-compatible. Existing callers and the default `execute()` path are unaffected.
- Only the DuckDB backend is modified. No changes to DataFusion or other backends.

### Negative

- **Cursor lifetime** — the cursor returned by `con.cursor()` must remain alive as long as the `RecordBatchReader` it produced is in use. Currently, the cursor is a local that could be garbage collected. In practice, DuckDB's `fetch_arrow_reader` detaches the result from the cursor, so the reader survives GC. If a future DuckDB version changes this behavior, readers from `isolated=True` calls may silently break.
- **Backend name check** — `register_and_transform_remote_tables` gates the `isolated` kwarg on `source_backend.name == "duckdb"`. If another backend (e.g., a future DuckDB-compatible engine) needs the same treatment, the check must be extended. An alternative would be a capability flag on the backend class, but this was deemed premature for a single backend.
- **Single call site** — `isolated=True` is currently used only in `register_and_transform_remote_tables`. If other code needs isolated DuckDB reads in the future, the pattern exists but is not documented beyond this ADR.
