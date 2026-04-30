# ADR-0009: Namespace S3 bucket layout by remote name and UUID

- **Status:** Accepted
- **Date:** 2026-04-28
- **Context area:** `python/xorq/catalog/annex.py`

## Context

A xorq catalog backed by `git-annex` stores content-addressed zip archives in an S3 (or GCS-via-S3-API) bucket via git-annex's S3 special remote. The bucket key for a given annex object is `{fileprefix}{annex-key}`, where `fileprefix` is configured at `initremote` time and stored in `remote.log` on the `git-annex` branch.

This raises two design questions that the original implementation did not answer:

1. **Multiple catalogs with the same name in one bucket.** Users want to operate several distinct catalogs (different git repos) that share a single bucket and may share a `name` (e.g. `my-first-catalog-annex`). With a flat fileprefix scheme, two same-named catalogs would write to the same path and clobber each other.
2. **Multiple clones of one catalog.** A catalog repo can be cloned many times. All clones must agree on where to read and write content; any per-clone fileprefix scheme breaks content-addressed dedup and risks `remote.log` mutation conflicts.

The original setup leaked the namespacing decision into shell tooling (`dev/init-catalog-submodule.sh` in xorq-gallery hand-built `fileprefix=annex-only/${uuidgen}/`), which produced orphan paths whenever the script's UUID generation diverged from what xorq later expected to read.

## Decision drivers

- Multiple catalogs sharing a bucket must not collide.
- All clones of one catalog must compute the same bucket path so content-addressed dedup works.
- The convention should be enforced inside xorq, not duplicated in every shell wrapper that touches a catalog.
- The `annex-uuid` sentinel that git-annex writes at `initremote` time should land at the *final* path, not at a parent path that subsequent operations abandon.

## Decision

xorq automatically composes `fileprefix` for annex-backed S3 remotes as:

```
{base-prefix}{remote-name}/{remote-uuid}/
```

Where:

- `{base-prefix}` is whatever the user supplied via `XORQ_CATALOG_S3_FILEPREFIX` (or constructed on an `S3RemoteConfig` directly). May be empty.
- `{remote-name}` is the git-annex special remote name (`name=` in `remote.log`).
- `{remote-uuid}` is the git-annex special remote's own UUID — the value in column 1 of the `remote.log` line for this remote, also stored in `.git/config` as `remote.<name>.annex-uuid`.

### Why the *remote* UUID and not the local-repo UUID

The remote UUID is generated once at `initremote` time and is stable across all clones of the catalog (it lives in `remote.log` on the shared `git-annex` branch). The local-repo UUID, by contrast, is per-clone — using it would mean every clone wrote to its own subdir, defeating content-addressed dedup and introducing a `remote.log` mutation race where `enableremote` calls from different clones append conflicting suffixes indefinitely.

### Pre-generating the remote UUID

`Annex.initremote(remote_config)` generates the UUID with `uuid.uuid4()` *before* invoking git-annex, then passes it to `git annex initremote` via the `uuid=<value>` parameter (a documented git-annex feature for forcing a specific UUID rather than letting git-annex generate one). This lets the suffix be baked into `fileprefix` from the very first call:

```python
remote_uuid = str(uuid.uuid4())
augmented = self._augment_fileprefix(remote_config, remote_uuid)
augmented.initremote(self.repo_path, uuid=remote_uuid)
```

Without pre-generation, the bucket would receive an `annex-uuid` sentinel at the parent path during initremote and a *second* sentinel at the final UUID-suffixed path on first content write — leaving an orphan.

### Idempotency

`S3RemoteConfig.augment_fileprefix(remote_uuid)` is a no-op when `self.fileprefix` already ends with `{name}/{remote_uuid}/`, so subsequent `enableremote` calls (e.g. xorq's `_try_resolve_annex_remote` runs one on every `Catalog.from_repo_path`) do not re-suffix. Across clones, the remote UUID is identical, so the idempotency check holds.

### Refusing to auto-migrate un-suffixed catalogs

`Annex.enableremote(rc)` reads the existing `fileprefix` for the remote out of `remote.log` and asks the config to verify it via `rc.verify_fileprefix(remote_uuid, existing_fileprefix)`. For S3, the verifier raises `AnnexError` when the existing prefix is not already namespaced by `{name}/{remote_uuid}/`. This is deliberate: silently rewriting `remote.log` on first open would mutate shared state on the `git-annex` branch and orphan bucket objects under the old prefix without warning. The error message points at ADR-0009 and tells the operator to copy bucket objects from the old prefix to one ending in `{name}/{remote_uuid}/` (e.g. `aws s3 cp --recursive`) and then update `remote.log` before reopening.

For `Directory` and `Rsync` remotes the verifier is a no-op — those backends already isolate via the directory or URL itself, so the namespacing question does not apply.

### Fallback in `enableremote`

`Annex.enableremote(rc)` looks up the remote UUID by name in `remote.log`. When the lookup misses, the fallback to `initremote` is *only* allowed when `remote.log` is empty — i.e. a genuinely fresh annex repo (`Catalog.init_repo_path(repo_path, annex=rc)`). A non-empty `remote.log` without a matching name raises `AnnexError`, so a typo'd remote name cannot accidentally create a parallel remote next to the existing one.

### Subclass interface change

`RemoteConfig.initremote(repo_path)` is extended to `initremote(repo_path, *, remote_uuid)`. The keyword is required: every call site goes through `Annex.initremote`, which always pre-generates the UUID via `uuid.uuid4()` and passes it down. Each subclass (`Directory`, `Rsync`, `S3`) emits `uuid={value}` in its CLI args.

### Where `augment_fileprefix` / `verify_fileprefix` live

Both methods live on `RemoteConfig` as no-op defaults, and are overridden on `S3RemoteConfig` only. `Annex.enableremote` and `Annex.initremote` dispatch through the config rather than introspecting fields, so adding a future remote type with its own namespacing scheme is a one-class change.

## Alternatives considered

### Use the local-repo annex UUID in the path

The first iteration of the patch used `git config --get annex.uuid` (the local repo's UUID) for the suffix.

Rejected because:
- Each clone of one catalog would have a different UUID and write to a different path, defeating dedup.
- `_try_resolve_annex_remote` calls `enableremote` on every `from_repo_path`, including by read-only consumers. With per-clone UUIDs, each call appends its own UUID to whatever `remote.log` already contained, causing `fileprefix` to grow without bound as clones take turns syncing.

### No UUID — namespace only by remote name

Use `{base-prefix}{remote-name}/`.

Rejected because:
- Two unrelated catalogs that happen to share a `name` (the stated user requirement) would collide on the same path.
- git-annex's drop-from-one-clone-drops-from-bucket model means the collision is destructive: a catalog dropping content would also remove that content for any other catalog at the same path.

### Two-phase initremote without `uuid=`

Run `initremote` with `fileprefix={base}{name}/`, read the assigned UUID out of `remote.log`, then `enableremote` with the suffixed `fileprefix`.

Deferred because:
- Leaves an orphan `annex-uuid` sentinel at the parent prefix on every fresh init.
- Requires an extra git-annex round-trip and an intermediate `remote.log` state that's easy to leak if the second step fails.

The `uuid=<value>` parameter on `initremote` is the supported alternative and produces the desired final state in one call.

## Consequences

### Positive

- Multiple catalogs with the same `name` cohabit a bucket without colliding (different remote UUIDs → different paths).
- Every clone of one catalog computes the same path → content-addressed dedup works → `git annex copy` and `get` resolve to the same bytes regardless of which clone wrote them.
- The convention lives entirely in xorq python; shell wrappers (e.g. `dev/init-catalog-submodule.sh` in xorq-gallery) no longer need to construct `fileprefix` themselves.
- The `annex-uuid` sentinel is written exactly once at the final path — no orphans.

### Negative

- **Existing catalogs whose `remote.log` carries a non-suffixed `fileprefix` cannot be opened until migrated.** `Annex.enableremote` raises `AnnexError` referencing this ADR; the operator must copy bucket objects from the old prefix to one ending in `{name}/{remote_uuid}/` (e.g. `aws s3 cp --recursive` or `gcloud storage cp -r`) and update `remote.log` before reopening. This is preferable to a silent auto-rewrite that would orphan objects in the bucket without warning.
- **`fileprefix` in `remote.log` no longer matches what the user typed in env vars.** Reading `remote.log` directly (e.g. for debugging) requires understanding that the value is the *resolved* path, not the *configured* base prefix.
- **The `uuid=<value>` initremote parameter ties xorq to a specific git-annex feature.** It is documented and stable, but if a future git-annex version removes it the patch would need to fall back to the deferred two-phase approach.

## References

- [git-annex S3 special remote — fileprefix](https://git-annex.branchable.com/special_remotes/S3/)
- [git-annex forum — When to reuse UUIDs and avoiding UUID clutter](https://git-annex.branchable.com/forum/When_to_reuse_UUIDs_and_avoiding_UUID_clutter/)
- ADR-0003 (optional git-annex backend) for the broader catalog-backend design this layers on.
