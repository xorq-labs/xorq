# ADR-0003: Make git-annex optional via a CatalogBackend abstraction

- **Status:** Accepted
- **Date:** 2026-03-24
- **Context area:** `python/xorq/catalog/annex.py`, `python/xorq/catalog/catalog.py`

## Context

The catalog stores versioned build artifacts (zip archives of serialized xorq expressions) in a git repository. Previously, git-annex was required: archives were content-addressed via annex, allowing clones to download only metadata and fetch artifact content on demand.

This created two problems:

1. **Deployment friction.** git-annex must be installed system-wide. Users who don't need lazy-fetch (local dev, CI, single-machine workflows) still pay this dependency cost.
2. **Tight coupling.** `Catalog` called git-annex operations directly, making it impossible to use a catalog without annex even when all content is local.

## Decision

Introduce a `CatalogBackend` abstract base class with two implementations:

- **`GitAnnexBackend`** — archives tracked by git-annex (symlinks to `.git/annex/objects`). Supports special remotes (S3, directory) for lazy content fetch.
- **`GitBackend`** — archives stored as plain git blobs. No external dependencies beyond git.

All `Catalog` classmethods (`from_repo_path`, `from_name`, `from_default`, `clone_from`, `from_kwargs`) accept an `annex` parameter to select the backend:

- `annex=None` (default) — plain git.
- `annex=LOCAL_ANNEX` — local-only git-annex (no special remote).
- `annex=<RemoteConfig>` — git-annex with a special remote (S3, directory, etc.).

### Interface

The ABC defines six operations that `Catalog` delegates to:

| Method | GitAnnexBackend | GitBackend |
|--------|----------|------------|
| `stage(path)` | `git add` | `git add` |
| `stage_content(path)` | `git annex add` + `git add` | `git add` |
| `stage_unlink(path)` | `git rm` + unlink | `git rm` + unlink |
| `commit_context(message)` | commit via index | commit via index |
| `is_content_local(path)` | checks symlink target resolves | `Path.exists()` |
| `fetch_content(path)` | `git annex get` on the entry | no-op |

`stage_content` is the key differentiator for writes: for annex it routes through `git annex add`; for plain git it's identical to `stage`. `is_content_local` and `fetch_content` support lazy content fetch: annex symlinks may exist without their target being present locally.

### What is NOT persisted

The backend type has no explicit field in `catalog.yaml`. However, annex remote configuration (S3 bucket, directory path, credentials spec) IS persisted in `catalog.yaml` under the `remote` key, and git-annex's own metadata lives in `.git/annex`. These serve as implicit signals for auto-detection if needed in the future.

### File layout

Both backends use an identical directory structure:

```
entries/<name>.zip                    # the archive (annex-tracked or plain blob)
metadata/<name>.zip.metadata.yaml    # sidecar metadata (always plain git)
aliases/<alias>.zip                   # symlink to ../entries/<name>.zip
catalog.yaml                         # entry/alias list + optional remote config
```

### Sidecar metadata

The sidecar file is always stored as a plain git blob (never annex-tracked), so it is available locally even when annex content has not been fetched. It contains:

```yaml
md5sum: "abc123..."
backends:
  - duckdb
expr_metadata:
  kind: source
  schema_out:
    col_a: Int64
    col_b: String
  schema_in: null
  root_tag: my_table
  composed_from:
    - entry_name: abc123
      alias: my-alias
      kind: source
```

- `md5sum` — integrity checksum of the archive.
- `backends` — backend connection names extracted from `profiles.yaml` in the archive.
- `expr_metadata` — the full `ExprMetadata.to_dict()` output (`kind`, `schema_out`, `schema_in`, `root_tag`, `parquet_cache_paths`, `composed_from`).

This means `CatalogEntry.metadata`, `.kind`, `.columns`, `.backends`, and `.composed_from` all read from the sidecar and never require the zip archive. Only `entry.expr` and `entry.lazy_expr` (which deserialize the full expression from `expr.yaml`) require fetching annex content.

## Rationale

### Why an ABC, not a flag inside Catalog?

Sprinkling `if self.annex:` throughout `Catalog` would couple the catalog logic to both storage strategies. The ABC keeps `Catalog` unaware of how staging works, and makes adding a third backend (e.g. DVC, LFS) a matter of implementing four methods.

### Why default to annex=None (plain git)?

The plain-git backend has zero external dependencies and covers the common case (local dev, CI, single-machine workflows). Users who need lazy-fetch via git-annex opt in by passing an `AnnexConfig` instance, which also carries the remote configuration — eliminating the invalid state of `annex=True` with a contradictory `remote_config`.

### Why not auto-detect the backend?

Auto-detection (check for `.git/annex` or `remote` key in `catalog.yaml`) was considered but deferred. The explicit `annex=` parameter is simpler and avoids surprising behavior when a user opens a repo that happens to have annex metadata from a previous experiment. Auto-detection can be added later as a convenience without changing the abstraction.

### Why an AnnexConfig type instead of a boolean?

A boolean `annex=True/False` plus a separate `remote_config` parameter creates invalid state combinations (e.g. `annex=False, remote_config=S3RemoteConfig(...)`). A single `annex` parameter that accepts `None | AnnexConfig` collapses both signals into one: the presence and type of the config object determines both *whether* to use annex and *how* to configure it.

`AnnexConfig` is the base; `RemoteConfig` (abstract) extends it and is further subclassed by `DirectoryRemoteConfig` and `S3RemoteConfig`. Passing `LOCAL_ANNEX` (a `LocalAnnexConfig`) gives local-only annex; passing a `RemoteConfig` subclass enables annex *with* a special remote. The hierarchy:

```
AnnexConfig
├── LocalAnnexConfig   (LOCAL_ANNEX singleton)
└── RemoteConfig (ABC)
    ├── DirectoryRemoteConfig
    └── S3RemoteConfig
```

## Guidelines for sidecar usage

The sidecar (`metadata/<name>.zip.metadata.yaml`) exists so that entry metadata is always available without fetching annex content. **New code should prefer sidecar-backed properties over deserializing the zip archive.**

### When to use the sidecar

Use `entry.metadata`, `entry.kind`, `entry.columns`, `entry.backends`, `entry.composed_from`, `entry.root_tag`, or `entry.sidecar_metadata` whenever you need metadata about an entry. These read from the git-tracked sidecar file and work even when annex content has been dropped.

### When `entry.expr` is required

Only access `entry.expr` or `entry.lazy_expr` when you need the deserialized expression itself — to execute it, build a `RemoteTable`, walk its operation graph, or pass it to `replace_unbound`. These require the zip archive and will raise `ContentNotAvailableError` if annex content is not local.

### Extending the sidecar

To promote a new field from the zip archive to the sidecar:

1. Add the field to `CatalogAddition.metadata` (in `catalog.py`), which builds the sidecar dict at add-time by reading from the `BuildZip`.
2. Expose the field on `CatalogEntry` as a property reading from `self.sidecar_metadata` or `self.metadata` (the `ExprMetadata` instance).
3. If the field belongs in `ExprMetadata`, add it to the `ExprMetadata` attrs class, `from_dict`, `from_expr`, and `to_dict` (in `vendor/ibis/expr/types/core.py`).
4. Update this ADR's sidecar YAML example.

Existing catalogs won't have the new field in their sidecar files. Handle missing keys with defaults (e.g. `data.get("new_field") or default`) so that old entries degrade gracefully.

## Consequences

### Positive

- Catalogs work without git-annex installed when `annex=None` (the default).
- Migration between backends is straightforward: `git annex add`/`git annex unannex` the archive files and reopen with the other backend. No schema changes needed.
- The identical file layout means a plain-git catalog can be upgraded to annex (or vice versa) without restructuring.

### Negative

- **Annex-only features require guards.** `set_remote_config` raises `NotImplementedError` on `GitBackend`; `get_remote_config` raises `NotImplementedError`. Callers that assume annex must check or catch.
- **No auto-detection.** Users must know which backend a repo uses and pass the correct config. A repo opened with the wrong backend will either fail (annex commands on a plain repo) or silently degrade (plain-git on an annex repo stores new entries as blobs alongside annex symlinks).
- **No early detection of missing git-annex.** `require_git_annex()` checks `shutil.which` at `Annex` construction and `init_repo_path` time, but a user who passes an `AnnexConfig` won't see the error until the first annex operation rather than at import time.

## Testing: S3 coverage and MinIO gaps

The `test_s3_remote_minio` test (`@pytest.mark.s3`) exercises the full git-annex S3 special remote round-trip — `initremote`, `copy --to`, `drop`, `get` — against a local MinIO instance. This validates the core plumbing but diverges from real AWS S3 in several ways.

### What MinIO covers

- git-annex special remote protocol (init, enable, copy, get, drop)
- Credential passing via `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars
- `embedcreds` serialization and `to_dict()` secret inclusion/exclusion
- `S3RemoteConfig` ↔ `catalog.yaml` round-trip

### Where MinIO diverges from AWS S3

| Area | MinIO test | Real AWS S3 |
|------|-----------|-------------|
| Signature version | v2 (hardcoded in `make_minio_remote`) | v4 (required since 2014 for new regions) |
| URL style | path-style (`http://host:9000/bucket`) | virtual-hosted (`https://bucket.s3.region.amazonaws.com`) |
| Protocol | `http`, no TLS | `https` with TLS negotiation |
| Authentication | static keys only | IAM roles, STS temporary credentials, instance profiles |
| Region routing | none — single local endpoint | region-specific endpoints with 301 redirects on mismatch |
| Storage classes | not meaningful | `STANDARD`, `GLACIER`, etc. affect retrieval latency and cost |
| Server-side encryption | `encryption=none` | SSE-S3, SSE-KMS, SSE-C |
| Network | localhost, sub-ms latency | real network with retries and timeouts |

### Gaps that could surface bugs in production

1. **Signature v4 + region routing.** `make_minio_remote` sets `signature="v2"`. Real AWS requires v4. git-annex needs the correct `datacenter`/`region` to construct the endpoint; a misconfigured region silently fails or triggers a 301 redirect that git-annex may not follow.

2. **IAM / STS credentials.** The `Annex.env` property explicitly clears `AWS_SESSION_TOKEN` and `AWS_SECURITY_TOKEN` (`annex.py:383-392`). This prevents temporary credentials from working — fine for MinIO, but means git-annex cannot use EC2 instance profiles or assumed roles.

3. **`enableremote` from a clone.** When a consumer clones the catalog and runs `enableremote`, auth failures and network errors surface differently against real S3 than against localhost MinIO. The `embedcreds=yes` path is especially important to validate end-to-end.

4. **`check_bucket()` via boto3.** The `S3RemoteConfig.check_bucket()` method (`annex.py:412-435`) validates credentials via boto3 before initializing the git-annex remote. This code path is never exercised in the MinIO test.

### Recommendation

Add a `@pytest.mark.aws` integration test that hits a real S3 bucket with `signature="v4"`, a non-trivial region, and virtual-hosted URL style. Gate it on `AWS_ACCESS_KEY_ID` presence so it only runs in CI environments with credentials configured. At minimum it should cover:

- `initremote` + `copy --to` + `get` round-trip with v4 signatures
- `enableremote` from a fresh clone (with and without `embedcreds`)
- `check_bucket()` validation before init
