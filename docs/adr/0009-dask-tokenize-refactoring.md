# ADR-0009: Move dask normalize functions into `__dask_tokenize__` class methods

- **Status:** Proposed
- **Date:** 2026-04-22
- **Context area:** `python/xorq/common/utils/dask_normalize/dask_normalize_expr.py`

## Context

xorq uses `dask.base.tokenize` to derive cache keys for every expression graph. The current design concentrates all normalization logic in a single 740-line module (`dask_normalize_expr.py`) that registers standalone functions via `@dask.base.normalize_token.register()`. Three problems arise:

**Discovery.** A reader of `rel.Read` or `rel.CachedNode` has no indication that a normalization function exists. Finding the rule requires knowing to look in `dask_normalize_expr.py` and scanning for the `@register` decorator — there is no co-location signal.

**Coupling.** `SnapshotStrategy` imports `normalize_backend`, `normalize_read`, and `normalize_remote_table` by name and re-exposes them as static methods. Any rename or refactor in `dask_normalize_expr.py` silently breaks the caching strategy.

**Brittleness.** The dispatch logic in `normalize_databasetable` and `normalize_backend` identifies backends by `dt.source.name` / `con.name` string literals. Every backend rename requires hunting down and updating scattered string guards across the file — and guards can be missed. The xorq backend has been renamed three times:

| Commit             | Rename                                    |
|--------------------|-------------------------------------------|
| `65d6857d` (#1450) | `"let"` → `"xorq"`                        |
| `fc4ff21c` (#1837) | `"xorq"` → `"xorq-datafusion"`            |
| `9db3be53` (#1851) | `"xorq-datafusion"` → `"xorq_datafusion"` |

Each rename required manual edits to `dask_normalize_expr.py`. Commit `125b9c42` (#1842) shows how this fails: `SnapshotStrategy.normalize_backend` in `strategy.py` missed the `"xorq-datafusion"` rename and required a separate fix four days later.

### How dask dispatches `normalize_token`

The `dask.base.normalize_token` object is a `Dispatch` instance. When `normalize_token(obj)` is called, it:

1. Walks `type(obj).__mro__` and checks `normalize_token._lookup` (the registered-function dict) for each class in the MRO.
2. If a match is found in `_lookup`, that function is called.
3. If no match is found anywhere in the MRO, falls through to `normalize_object` (registered for `object`).
4. `normalize_object` checks `obj.__dask_tokenize__` and calls it if present.

Consequence: a `_lookup` entry (including one injected by `patch_normalize_token`) **always takes priority over `__dask_tokenize__`**. This means `SnapshotStrategy.normalization_context` continues to work correctly after the migration, because the context manager patches `_lookup` directly.

### Vendored ibis

xorq maintains a forked copy of ibis under `python/xorq/vendor/ibis/`. This copy is already diverged from upstream — it carries xorq-specific operations (`Tag`, `HashingTag`, `FlightExpr`, etc.) that will never be upstreamed. Upstream ibis have a steady, low-pace cadence that makes tracking manageable, particularly with LLM-assisted diff review. Adding `__dask_tokenize__` to vendored classes is equivalent to any other targeted modification of the vendor layer.

## Decision drivers

- Normalization logic must be co-located with the type it normalizes.
- The `SnapshotStrategy` patching mechanism (`patch_normalize_token` via `_lookup`) must continue to work without modification.
- Backend renames must not require edits to `dask_normalize_expr.py` or any centralized dispatch table.
- Migration should be incremental and independently testable.
- `normalize_op` must remain importable from its current module path — `patch_normalize_op_caching` patches it by module attribute reference.

## Decision

### Core principle

Every type with a standalone registered function gets a `__dask_tokenize__` method. For types that dispatch to backend-specific logic (specifically `ir.DatabaseTable` and `ibis.backends.BaseBackend`), the dispatch is pushed all the way to each backend class — eliminating any centralized string-keyed or instance-checked dispatch table.

### `DatabaseTable` and the `tokenize_table` protocol

`ir.DatabaseTable.__dask_tokenize__` delegates entirely to the backend:

```python
def __dask_tokenize__(self):
    return self.source.tokenize_table(self)
```

`BaseBackend` gains a `tokenize_table(self, dt)` protocol method (raises `NotImplementedError` by default). Each xorq backend class in `python/xorq/backends/` implements it with the logic currently in the corresponding `normalize_X_databasetable` function. There is no dispatch dict, no string comparison, no `isinstance` check — Python's method resolution handles dispatch.

`BaseBackend.__dask_tokenize__` moves the connection-identity logic from `normalize_backend` into the base class, with each backend overriding as needed.

### Target classes

**xorq-owned** (`python/xorq/expr/` and `python/xorq/expr/`):

| Standalone function                | Class                       | File                             |
|------------------------------------|-----------------------------|----------------------------------|
| `normalize_read`                   | `rel.Read`                  | `python/xorq/expr/relations.py`  |
| `normalize_remote_table`           | `rel.RemoteTable`           | `python/xorq/expr/relations.py`  |
| `normalize_cached_node`            | `rel.CachedNode`            | `python/xorq/expr/relations.py`  |
| `normalize_named_scalar_parameter` | `xops.NamedScalarParameter` | `python/xorq/expr/operations.py` |
| `normalize_ibis_datatype`          | `dat.DataType`              | `python/xorq/expr/datatypes.py`  |

**Vendored ibis** (`python/xorq/vendor/ibis/`):

| Standalone function       | Class                  | File                                                   |
|---------------------------|------------------------|--------------------------------------------------------|
| `normalize_schema`        | `ir.Schema`            | `python/xorq/vendor/ibis/expr/schema.py`               |
| `normalize_namespace`     | `ir.Namespace`         | `python/xorq/vendor/ibis/expr/operations/relations.py` |
| `normalize_scalar_udf`    | `ScalarUDF`            | `python/xorq/vendor/ibis/expr/operations/udf.py`       |
| `normalize_agg_udf`       | `AggUDF`               | `python/xorq/vendor/ibis/expr/operations/udf.py`       |
| `normalize_expr`          | `ibis.expr.types.Expr` | `python/xorq/vendor/ibis/expr/types/core.py`           |
| `normalize_databasetable` | `ir.DatabaseTable`     | `python/xorq/vendor/ibis/expr/operations/relations.py` |
| `normalize_backend`       | `BaseBackend`          | `python/xorq/vendor/ibis/backends/__init__.py`         |

**Backend `tokenize_table` implementations** (`python/xorq/backends/`):

| Current function                     | Backend class             | File                                                      |
|--------------------------------------|---------------------------|-----------------------------------------------------------|
| `normalize_pandas_databasetable`     | `pandas.Backend`          | `python/xorq/backends/pandas/__init__.py`                 |
| `normalize_datafusion_databasetable` | `datafusion.Backend`      | `python/xorq/backends/datafusion/__init__.py`             |
| `normalize_postgres_databasetable`   | `postgres.Backend`        | `python/xorq/backends/postgres/__init__.py`               |
| `normalize_snowflake_databasetable`  | `snowflake.Backend`       | `python/xorq/backends/snowflake/__init__.py`              |
| `normalize_xorq_databasetable`       | `xorq_datafusion.Backend` | `python/xorq/backends/xorq_datafusion/__init__.py`        |
| `normalize_duckdb_databasetable`     | `duckdb.Backend`          | `python/xorq/backends/duckdb/__init__.py`                 |
| `normalize_remote_databasetable`     | `trino.Backend`           | `python/xorq/backends/trino/__init__.py`                  |
| `normalize_remote_databasetable`     | `gizmosql.Backend`        | `python/xorq/backends/gizmosql/__init__.py`               |
| `normalize_bigquery_databasetable`   | `bigquery.Backend`        | `python/xorq/vendor/ibis/backends/bigquery/__init__.py` ¹ |
| `normalize_pyiceberg_database_table` | `pyiceberg.Backend`       | `python/xorq/backends/pyiceberg/__init__.py`              |
| `normalize_sqlite_database_table`    | `sqlite.Backend`          | `python/xorq/backends/sqlite/__init__.py`                 |

¹ `bigquery` has no xorq-owned subclass; `tokenize_table` is added directly to the vendored ibis backend.

### Out of scope

**`normalize_module`.** `types.ModuleType` is a stdlib type; adding a method to it is not possible. The `@normalize_token.register(types.ModuleType)` decorator stays.

**`normalize_op` and `opaque_node_replacer`.** Internal helpers called by `normalize_expr`, not registered for specific types. They remain as module-level helpers in `dask_normalize_expr.py`. `patch_normalize_op_caching` patches `normalize_op` by module attribute reference and must continue to find it at `xorq.common.utils.dask_normalize.dask_normalize_expr.normalize_op`.

### `SnapshotStrategy` import update

After migrating, update `python/xorq/caching/strategy.py`:

- Remove the three named imports (`normalize_backend`, `normalize_read`, `normalize_remote_table`).
- `cached_normalize_read`: `normalize_read(op)` → `dask.base.normalize_token(op)`.
- `normalize_databasetable`: `normalize_remote_table(dt)` → `dask.base.normalize_token(dt)`.
- `normalize_backend` fallback: `normalize_backend(con)` → `dask.base.normalize_token(con)`.

## Alternatives considered

### Keep all normalize functions as registered module-level functions

Rejected: normalization logic remains invisible from the class definition, `SnapshotStrategy` stays coupled to internal helper names, and every backend rename forces edits to `dask_normalize_expr.py`.

### Replace string dispatch with `isinstance`-based dispatch in `normalize_databasetable`

Rejected: still a centralized dispatch table, still requires editing one file when a new backend is added. Pushing dispatch to the backend class eliminates the centralized table entirely.

### Exclude vendored-ibis classes from the migration

Deferred initially on vendor-maintenance grounds. Rejected on reexamination: xorq's vendor copy is already a diverged fork; adding `__dask_tokenize__` and `tokenize_table` is no different from existing targeted modifications.

## Consequences

### Positive

- Normalization logic is co-located with the type it describes — a reader of `postgres.Backend` sees `tokenize_table` directly.
- Adding a new backend only requires implementing `tokenize_table` on the new backend class; `dask_normalize_expr.py` is not touched.
- Backend renames do not affect tokenization code at all.
- `SnapshotStrategy` no longer imports named helpers from `dask_normalize_expr.py`.
- `dask_normalize_expr.py` shrinks to `normalize_op`, `opaque_node_replacer`, `normalize_module`, `normalize_inmemorytable`, and their private helpers.

### Negative

- Vendored ibis files diverge further from upstream. Accepted: upstream pace is low and LLM-assisted diff review makes periodic reconciliation tractable.
- `bigquery` lacks a xorq-owned `Backend` subclass; its `tokenize_table` goes into the vendored copy. A future xorq-owned bigquery subclass would be the natural resolution.
- `normalize_op` must stay at its current module path.

## References

- `python/xorq/common/utils/dask_normalize/dask_normalize_expr.py` — current normalize functions
- `python/xorq/caching/strategy.py` — `SnapshotStrategy.normalization_context` and named imports
- `python/xorq/expr/relations.py` lines 119–160 — `Tag` and `HashingTag.__dask_tokenize__` (existing template)
- Commits `65d6857d`, `fc4ff21c`, `9db3be53`, `125b9c42` — backend rename churn in `dask_normalize_expr.py`
- dask 2025.1.0 `dask/tokenize.py` — `normalize_object` shows `__dask_tokenize__` is checked only when no `_lookup` entry exists
