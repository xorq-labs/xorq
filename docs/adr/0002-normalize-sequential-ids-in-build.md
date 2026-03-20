# ADR-0002: Normalize sequential IDs during expression builds for deterministic hashing

- **Status:** Accepted
- **Date:** 2026-03-19
- **Context area:** `python/xorq/ibis_yaml/compiler.py`, `python/xorq/common/utils/graph_utils.py`

## Context

`build_expr` serializes an Ibis expression to YAML and computes a content hash that determines the build directory. The hash and the file content must be deterministic: the same logical expression must always produce the same build artifacts, regardless of the order in which connections, UDFs, and other objects were created during the session.

Several objects in the expression graph carry session-global sequential IDs:

| Source | Mechanism | Affects hash? | Affects YAML? |
|--------|-----------|:---:|:---:|
| `Profile.idx` | `itertools.count().__next__` on each `Profile` instantiation | Yes (via `hash_name` in `dehydrate_cons`, `translate_to_yaml`, `get_expr_hash`) | Yes (profile keys, `profile:` / `source:` fields) |
| UDF `__class__.__name__` | `_make_udf_name` appends `_N` counter per function name | No (`normalize_scalar_udf` ignores class name) | Yes (`class_name:` field in UDF YAML) |
| `Reference.identifier` | `itertools.count()` on `SelfReference` / `JoinReference` | No (SQL compilation and `normalize_op` are insensitive) | Yes (`identifier:` field) |
| `_count` in `register_and_transform_remote_tables` | `itertools.count()` for temp table names | No (execution-time only) | No |
| `gen_name` UUID names | `uuid4()` on `InMemoryTable` / `Read` nodes | Would, but already handled | Would, but already handled by `_sanitize_generated_names` |

Creating the same connections in a different order produces different `Profile.idx` values, which changes both the build hash and every YAML file that references a profile. This makes builds non-reproducible across sessions.

## Decision

Normalize sequential IDs at build time so that both the hash and file content are deterministic.

### Profile.idx: graph rewrite via `replace_sources`

A new general-purpose function `replace_sources(source_mapping, expr)` in `graph_utils.py` rewrites an expression graph, replacing backend references. It handles:

- `node.source` on all source-bearing ops (`DatabaseTable`, `Read`, `RemoteTable`, `CachedNode`, `FlightExpr`, `FlightUDXF`, `SQLQueryResult`)
- `node.cache.storage.source` on `CachedNode` ops (nested inside frozen attrs objects, rebuilt via `attr.evolve`)
- Recursive descent into opaque sub-expressions via the existing `replace_nodes` infrastructure

`normalize_profiles(expr)` builds on `replace_sources` to:
1. Collect all backends from the expression via `find_all_sources`
2. Sort them by content-only hash (idx excluded) — Python's stable sort preserves discovery order for ties
3. Assign canonical idx = 0, 1, 2, … in sorted order
4. Shallow-copy each backend that needs a new idx, assign the canonical profile, and explicitly share `backend.con` so registered tables and session state are preserved
5. Rewrite the graph via `replace_sources`

This runs in `ExprDumper.__attrs_post_init__` before the hash is computed, so both the hash and all serialized YAML see the canonical values. The original expression and its backends are not mutated.

### UDF class_name: use `__func_name__` instead of `__class__.__name__`

All three UDF YAML serializers (`ScalarUDF`, `AggUDF`, `ExprScalarUDF`) now emit `__func_name__` (the user-given function name, e.g. `my_udf`) as `class_name` rather than `__class__.__name__` (which includes the counter suffix, e.g. `my_udf_0`). The `from_yaml` path uses `class_name` only to name the dynamically reconstructed type — the actual function comes from the pickle, so this is safe.

### Reference.identifier: not normalized

`Reference.identifier` values are serialized into YAML but do not affect the build hash (SQL compilation and `normalize_op` are insensitive to them). More importantly, identifiers disambiguate self-joins — two references to the same table in a self-join must have distinct identifiers to be resolved correctly. Normalizing them would require understanding the join structure to preserve correctness. Since they don't affect the hash, they are left as-is.

## Rationale

### Why graph rewrite, not mutation?

An earlier iteration mutated `backend._profile` in place. This is simpler but leaks: backends are shared objects, so mutating a profile during `build_expr(A)` also changes what expression B sees if it references the same backend. A context manager (mutate-then-restore) was considered but rejected in favor of the graph rewrite because:

- Graph rewrites are pure — no risk of leaking state to other expressions
- `replace_sources` is a reusable primitive for future connection replacement needs (migration, testing, etc.)
- The shallow-copy + shared `.con` pattern is well-understood and testable

### Why shallow copy + shared `.con`?

`copy(backend)` alone does not share the underlying connection engine — some backends (xorq, duckdb) re-invoke `do_connect` during copy, producing an empty connection without registered tables. Explicitly assigning `cloned.con = backend.con` after the copy ensures the clone sees the same session state. Not all backends have a `.con` attribute (e.g. pandas), so the assignment is conditional.

### Why sort by content hash?

Content hash sort produces a canonical ordering that is independent of:
- The order in which connections were created in the session
- The order in which they appear in the expression graph
- Whether the expression was built incrementally or all at once

Python's stable sort ensures that backends with identical content (e.g. multiple xorq backends with the same config) preserve their discovery order from `find_all_sources`, which is itself deterministic (BFS traversal order).

## Consequences

### Positive

- `build_expr` produces identical hashes and file content for the same logical expression regardless of session state.
- `replace_sources` is available as a general-purpose primitive for any future need to swap backends in an expression graph (testing, migration, multi-environment deployment).
- No mutation of shared state — safe for concurrent or sequential builds of expressions sharing backends.

### Negative

- **Shallow copy fragility** — the `copy(backend) + shared .con` pattern assumes that the only mutable state that matters for query execution is on `.con`. If a backend stores execution-relevant state elsewhere (e.g. in instance variables set after `do_connect`), the clone may behave incorrectly. New backend implementations should be tested with `_clone_backend_with_profile`.
- **Sort stability assumption** — if `find_all_sources` traversal order changes (e.g. due to graph representation changes), the canonical idx assignment for same-content backends will change, invalidating existing build caches.
- **Snapshot test churn** — build stability snapshot tests needed updating since the canonical idx values differ from the manually assigned values used previously.
