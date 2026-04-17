# ADR-0007: Canonicalize catalog-extract tempdir in DataFusion DT tokens

- **Status:** Accepted
- **Date:** 2026-04-17
- **Context area:** `python/xorq/common/utils/dask_normalize/dask_normalize_expr.py`, `python/xorq/ibis_yaml/compiler.py`
- **Related:** ADR-0006 (covers the parallel concern at the `Read` op level)

## Context

`normalize_datafusion_databasetable` tokenizes a DataFusion-backed `DatabaseTable` by parsing absolute file paths out of the backend's `execution_plan()` string. The plan string contains whatever absolute paths the backend registered the table under.

`load_expr_from_zip` picks a fresh `tempfile.mkdtemp(prefix="xorq-catalog-")` per load and extracts the build bundle there. Two loads of the same catalog entry therefore see the same files at *different* absolute paths, and the raw plan-string token differs per load — the same failure mode ADR-0006 describes for the `Read` layer, but one layer down.

ADR-0006 fixes the `Read`-op side by splitting `read_kwargs` into `hash_path` (absolute, for tokenization) and `read_path` (relative to the build root, for I/O). That split does not reach `normalize_datafusion_databasetable`, which does not look at `read_kwargs` at all — it reads the backend's execution plan, which is already bound to whatever path the loader used at register time.

A prior attempt (`ce8004bc`, on `main`) replaced the plan-path token with content-md5 digests of the underlying files, cached on the backend via `stash_datafusion_content_digests`. That stabilized tokens across reloads but broke the `same-path-same-token` cache contract that `test_parquet_cache_storage` pins: a cache entry is expected to survive a content-change at the same path (schema change is the documented invalidator, not content change). Content-digest tokens invalidated on every content rewrite.

Any fix has to reconcile:

- Stability across `load_expr_from_zip` tempdirs (the bug this ADR addresses).
- The `same-path-same-token` semantic (the contract ADR-0006 also names as load-bearing).
- Distinguishability across genuinely different files.

Content-digest satisfies 1 and 3 but violates 2. Raw plan paths satisfy 2 and 3 but violate 1.

## Decision

Keep the plan-path-based token, but canonicalize paths by stripping the catalog extract-dir prefix before tokenization. `_CATALOG_EXTRACT_DIR_RE = re.compile(r".*?/xorq-catalog-[^/]+/")` captures everything up through the first `xorq-catalog-<random>/` segment; `_canonicalize_plan_path` substitutes it out.

After stripping, the remaining suffix begins at the build-hashed subdir (`<build-hash>/memtables/foo.parquet`, `<build-hash>/database_table/bar.parquet`, etc.), which is stable across reloads because the build hash is content-addressed.

Non-catalog paths — user-supplied files that never pass through `load_expr_from_zip` — don't match the regex and pass through unchanged. They keep their absolute paths, and `same-path-same-token` holds.

`_extract_plan_file_paths` returns a sorted `tuple` of canonicalized path strings; `normalize_datafusion_databasetable` feeds that tuple plus `dt.schema.to_pandas()` into `normalize_seq_with_caller`.

### Why canonicalize the path rather than hash contents

Three reasons, mirroring ADR-0006's "Why `hash_path` is absolute":

- **Preserves the cache contract.** `test_parquet_cache_storage` requires that a cached expression continue to hit the cache after the source parquet at the same path is overwritten. Path-based tokens give that for free; content-digest tokens don't.
- **No I/O at tokenization time.** Content-digest requires opening every registered file during normalization. For large catalogs this is expensive and has to be pre-stashed (the removed `stash_datafusion_content_digests`) to avoid paying the cost per token call.
- **Consistency with ADR-0006.** The `Read` layer hashes file *contents* via `normalize_read`; the DT layer operates on what the backend already registered. Doing content-digest at both layers would double-hash the same bytes; doing it only at the DT layer makes the two layers inconsistent in a non-obvious way.

### Why regex-strip rather than a structured extract-root carrier

A cleaner-looking alternative is to stash the extract root on the backend (or thread it through a context) and strip it symbolically. Rejected because:

- The plan string is produced by DataFusion, not by us. Any structured approach still has to parse the string to find paths, then match them against a known prefix. The regex is doing the matching either way.
- The `xorq-catalog-<random>/` prefix shape is stable and owned by us (`tempfile.mkdtemp(prefix="xorq-catalog-")` in the catalog loader). A regex keyed on that shape is as precise as a structured match, with fewer moving parts.
- There is no second consumer that would benefit from a structured extract-root accessor. Introducing one now would be speculative.

## Consequences

- **DT-token form changes again.** `main` shipped `ce8004bc`'s content-digest tuple; this ADR returns to a path-tuple token (canonicalized). Any token computed under `ce8004bc` is now non-equivalent. Snapshot tests updated in the same commit (`test_tokenize_datafusion_parquet_expr`). Downstream caches keyed on DT tokens are invalidated on upgrade. Acceptable for the same reason ADR-0006 accepted YAML forward-incompatibility: the cache key is not a stability surface we promise across versions.
- **`stash_datafusion_content_digests` removed.** `ExprLoader.deferred_read_to_memtable` no longer stashes digests on the backend after `make_dt`, and the `_xorq_content_digests` attribute is no longer set anywhere. The backend-side digest cache is gone.
- **Regex coupling to `tempfile.mkdtemp` prefix.** If the catalog loader ever changes its extract-dir prefix, the regex must change in lockstep. Named constant + single call site makes that a one-line change, but it is a coupling worth flagging.

## Alternatives considered

- **Content-digest at the DT layer (the reverted `ce8004bc` approach).** Rejected: violates `same-path-same-token`, requires pre-stashing to avoid per-token I/O, and double-hashes content that `normalize_read` already hashes.
- **Structured extract-root carrier on the backend.** Rejected: adds surface area without removing the string parsing step, and has no second consumer.
- **Defer tokenization until the backend exposes a stable path accessor.** Rejected: no such accessor is on any roadmap, and the extract-dir variance is already causing test and cache failures today.
- **Rebase the DT path on `hash_path` from ADR-0006.** Rejected: `normalize_datafusion_databasetable` dispatches on op type and receives only the `DatabaseTable`. There is no back-reference from a registered DT to an originating `Read` — and a DT need not have one, since backends can register tables directly without going through a `Read`. Recovering the link would require either changing the normalizer's signature to take the enclosing expression, or stamping a `Read`-id onto the DT at `make_dt` time. Both widen the surface across `Read` → `make_dt` → normalize to solve a problem the regex handles at one call site.
