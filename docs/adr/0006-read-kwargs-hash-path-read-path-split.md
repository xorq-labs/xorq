# ADR-0006: Split `read_kwargs` path into `hash_path` (identity) and `read_path` (location)

- **Status:** Accepted
- **Date:** 2026-04-17
- **Context area:** `python/xorq/expr/relations.py`, `python/xorq/ibis_yaml/compiler.py`, `python/xorq/common/utils/defer_utils.py`, `python/xorq/common/utils/dask_normalize/dask_normalize_expr.py`

## Context

A `Read` op records how to rebuild a DataFusion-registered table from a file on disk. The file reference has two distinct jobs:

1. **Identity for hashing.** `normalize_read` feeds a path into a dask token so that two Reads of the same file produce the same token, and thus hit the same cache. For local files this is an md5 of contents; for remote (`s3://`, `gs://`) it is object metadata.
2. **Location for I/O.** `make_dt` and `deferred_reads_to_memtables` open the file to register it with the backend at load time.

Before this ADR, a single `"path"` key in `read_kwargs` served both jobs. That worked when the file sat at the same absolute path for the life of the expression, but broke on catalog roundtrips:

- `build_expr` writes the inline memtable parquet into the build tmpdir and stores its absolute path as `"path"`.
- `Catalog.add` zips that build dir; `Catalog.load` extracts to a *different* tmpdir (`.../xorq-catalog-<random>/<build-hash>/memtables/foo.parquet`).
- `expr_path.joinpath(stored_path)` discards `expr_path` because `stored_path` is absolute. The loaded expression points back at a build tmpdir that no longer exists.

Any fix has to reconcile two constraints that point in opposite directions:

- The hashing contract depends on a *stable, absolute* identifier. Changing the identifier (e.g., rewriting to a relative path at build time) invalidates caches and breaks the `same-path-same-token` semantic that `test_parquet_cache_storage` relies on (cache invalidates on schema change, not on content change at the same path).
- The I/O contract needs a path *relative to the current extract root* so the loader can resolve it wherever the zip happens to be extracted.

One key cannot satisfy both.

## Decision

Split `read_kwargs` into two path keys with disjoint responsibilities:

| Key | Value | Consumers | When present |
|---|---|---|---|
| `hash_path` | absolute path at build time (or remote URL) | `normalize_read`, `_transform_deferred_reads` in `expr/api.py` | every Read |
| `read_path` | path relative to the build root, e.g. `inmemory/<uuid>.parquet` or `database_table/<uuid>.parquet` | `ExprLoader.deferred_read_to_memtable` (`expr_path.joinpath(read_path)`), `Read.make_dt` via exclusion list | only on Reads emitted by the build compiler for inline memtables and `database_table` memtables |

`make_read_kwargs` in `defer_utils.py` normalizes every backend-specific path parameter (`path`, `paths`, `source`, `source_list`) into `hash_path` — one spelling for one job, so `normalize_read` no longer needs the `try_names` fallback.

### Why `read_path` is relative, not absolute

`expr_path.joinpath(read_path)` is the load-time resolution. For a relative `read_path`, this produces `<current-extract-root>/<relative>` — portable across extracts. An absolute `read_path` would be discarded by `joinpath`, recreating the original bug.

### Why `hash_path` is absolute, not relative

The hash contract must survive rebuilds at unrelated paths *without* changing the token, and it must also distinguish files at different paths. An absolute path satisfies both — rebuilds of the same content produce the same md5, and different files at different paths tokenize differently. The fact that two rebuilds of the same content at *different* absolute paths produce different absolute `hash_path` strings is fine: `normalize_read` for local files hashes *file contents*, not the path string. The path just tells it *which file to read*.

### Why not compute `read_path` from `hash_path` at load time

Tempting, but wrong:

- `hash_path` can be a remote URL (`s3://...`) for which there is no local file — `read_path` would not apply.
- Some Read nodes (e.g., direct `deferred_read_parquet` against a persistent file) have `hash_path` but no `read_path` because they don't live under the build root; the loader must not try to join them.

The presence or absence of `read_path` is the signal: "this Read was materialized into the build bundle and must be joined to the extract root" vs "this Read is a user-supplied file that already lives where it says."

`ExprLoader.deferred_read_to_memtable` uses exactly this signal:

```python
drs = tuple(
    dr for dr in walk_nodes(Read, loaded)
    if "read_path" in dict(dr.read_kwargs)
)
```

### `make_dt` exclusion list

When a Read is executed directly (not materialized through the loader's memtable path), `make_dt` calls the backend's read method with `hash_path` as the positional argument and forwards the rest as kwargs. Both `hash_path` and `read_path` must be stripped from the forwarded kwargs — `hash_path` becomes the positional argument, and `read_path` is meaningful only to the loader. The exclusion list in `Read.make_dt` enumerates both.

## Consequences

- **YAML schema change.** Build artifacts written before this ADR use `"path"`; after, they use `"hash_path"` (and optionally `"read_path"`). Three `test_build_file_stability_*` snapshots were updated in the same commit. Old catalogs with pre-split YAML are not forward-compatible — they would fail `normalize_read`'s `read_kwargs["hash_path"]` lookup. We accept this because catalog entries are already tied to build hashes that embed the YAML schema.
- **Name clarity at call sites.** A reader who sees `hash_path` in `normalize_read` or `read_path` in `deferred_read_to_memtable` immediately knows which job is being done. Before the split, the same `"path"` key appeared in both contexts and the intent had to be inferred from the surrounding function.
- **Dead code removed.** `normalize_read`'s `try_names = ("path", "paths", "source", "source_list")` fallback is gone — `make_read_kwargs` now guarantees a single spelling.

## Alternatives considered

- **Store the path relative everywhere.** Rejected: breaks the `same-path-same-token` cache semantic and invalidates every existing cache on upgrade.
- **Rewrite `"path"` to relative at load time, to absolute at build time.** Rejected: forces every reader of `read_kwargs` to know which half of the lifecycle it's in. The key/value contract becomes time-dependent, which is hard to reason about and easy to break.
- **Keep a single `"path"` and special-case `Path.joinpath` to prepend `expr_path` even for absolute paths.** Rejected: `joinpath`'s absolute-path semantics are a feature (the user can always override), not a bug worth working around at one call site.
