# ADR-0009: Catalog supports a single git remote

- **Status:** Proposed
- **Date:** 2026-04-30
- **Context area:** `python/xorq/catalog/catalog.py`

## Context

A `Catalog` is a git repository — usually with a git-annex sidecar branch — that holds versioned build artifacts. `Catalog.push()`, `Catalog.pull()`, and `Catalog.fetch()` operate over `self._git_remotes`, which is *every* git remote on the underlying repo that has a fetch refspec (see the `Catalog._git_remotes` property in `python/xorq/catalog/catalog.py`). Nothing in the catalog API restricts a user to one remote; if `git remote add r2 …` is run on the working tree, the next `catalog.push()` will dutifully push to both `origin` and `r2`.

Issue [#1898](https://github.com/letsql/xorq/issues/1898) surfaced a silent failure mode in this design: when a push to one remote was rejected (non-fast-forward), `catalog.push()` returned without raising. The fix landed in three commits on this branch:

| Commit | Change |
|--------|--------|
| `fe3f608b` | Raise `CatalogPushError` when `PushInfo` flags include `REJECTED`/`REMOTE_REJECTED`/`REMOTE_FAILURE`/`ERROR` |
| `6b2ab300` | Push every ref to every remote first, then aggregate all rejections into one error message — instead of bailing on the first failure |
| `0b40bb35` | Test covering two-remote partial-failure aggregation |

The aggregation fix exists because the obvious "fail fast on the first rejected push" implementation has a worse failure mode than the bug it replaces: with two git remotes, rejecting on `origin` skips the push of `origin`'s annex branch *and* the entire push to `r2`, so a retry sees inconsistent state across the two remotes. Aggregation patches that — but it patches *one* of an open-ended set of multi-remote consistency problems we have not enumerated:

- **Atomicity (between remotes).** `push()` to N remotes is N independent operations. Any non-empty subset can succeed while the rest fail, leaving remotes out of sync. Aggregating the error tells the user *that* this happened, not how to recover. Note: single-remote does not give us atomicity either — `main` and `git-annex` are pushed in two separate operations inside `Catalog.push`, so even one remote can be left half-pushed. Single-remote shrinks the inconsistency surface from `2N` operations to `2`; it does not eliminate it.
- **Annex content placement.** `git-annex` tracks per-key location across remotes. With multiple git remotes plus a special remote, "where does the content live" stops being a yes/no question. `Catalog.fetch_entries` just calls `backend.fetch_content`; it does not reason about which remote satisfies which key.
- **Pull semantics.** `Catalog.pull` calls `Remote.pull` on each remote in turn. With divergent histories on two remotes, the first pull succeeds and the second fails (or merges) silently with respect to the catalog's consistency invariants.
- **`assert_consistency` against which remote?** Mutating operations (`add`, `remove`, `add_alias`) wrap their work in `Catalog.synchronizing`, which pulls from every remote then pushes to every remote; `assert_consistency` runs inside the wrapped block (e.g. `Catalog._add_zip`). With multiple remotes, "the catalog is consistent with the remote" is undefined — we may have pulled from `origin` and pushed to `r2`, with `r1` untouched.
- **`remote_config` is singular.** The git-annex `RemoteConfig` resolution in `_try_resolve_annex_remote` reads one config from `remote.log`. The data model already assumes a single source of truth for credentials and bucket location.

We have no users today asking for multi-remote, no design for what the operations above *should* do, and no test coverage beyond the push-failure aggregation case. The aggregation fix is correct given the current API, but committing to it as a supported surface area locks in design debt we have not paid down.

The aggregation fix landed before this ADR rather than alongside it because it is a strict improvement over silent rejection regardless of what we decide about multi-remote: the single-remote case is also better-served by raising than returning successfully on a rejected push. Writing the aggregation logic is what surfaced the larger surface-area question this ADR resolves.

## Decision drivers

- The catalog API is small and recent; we have room to constrain it before users build workflows that depend on multi-remote behavior.
- Single-remote is the only configuration we test, document, or run in production.
- Failure modes that "look like they work" are worse than failure modes that raise loudly.
- We want to hear from users with this need rather than guess at requirements.

## Decision

The catalog officially supports exactly one git remote. Configurations with two or more git remotes are unsupported, and the catalog will refuse to operate on them rather than attempting best-effort multi-remote semantics.

### What "supported" means

- `Catalog.push()`, `Catalog.pull()`, `Catalog.fetch()`, and `Catalog.sync()` are defined for repos with zero or one git remote.
  - **Zero remotes** is the local-only case (used in tests and by `init=True` before a remote is wired up). These operations are no-ops in that mode and do not raise. We accept the silent no-op here — unlike the rejected-push case, the user has not configured a remote, so there is no "I asked for X and got nothing" surprise. Local-only workflows (development, tests, fixture setup) want this.
  - **One remote** is the supported configuration; behavior is unchanged from today minus the multi-remote aggregation logic.
- A repo with two or more git remotes raises `CatalogConfigurationError` the first time one of those operations is invoked, with a message that names the remotes found and points the user at the single-remote constraint. `CatalogConfigurationError` is a new `RuntimeError` subclass defined in `python/xorq/catalog/catalog.py` alongside `CatalogPushError`, and is exported from `xorq.catalog`.
- `Catalog.clone_from`, `Catalog.add`, `Catalog.remove`, and read-only operations are unaffected — they do not depend on a remote.

### What "git remote" means here

This ADR constrains *git* remotes — entries in `repo.remotes` with a fetch refspec, as enumerated by `Catalog._git_remotes`. The git-annex *special* remote (the S3 / directory / etc. configured via `RemoteConfig` and stored in `remote.log`) is unrelated and remains singular as it always has been. A catalog continues to look like:

```
catalog repo  ──push/pull──>  one git remote (e.g. github.com/org/catalog.git)
     │
     └── annex content  ──get/copy──>  one annex special remote (e.g. s3://bucket)
```

### Implementation sketch

Add a `_assert_at_most_one_git_remote` guard called at the top of `push`, `pull`, `fetch`, and `sync`. (Putting it inside `synchronizing` is tempting but insufficient: `push`/`pull`/`fetch` are public methods callers invoke directly without going through `synchronizing`, and `sync` enters `synchronizing` *itself*, so the guard must run before the context manager body.) The guard raises `CatalogConfigurationError` when `len(self._git_remotes) > 1` and is a no-op otherwise. The aggregation loop in `Catalog.push` collapses to a single-remote push-and-collect.

Add a `Catalog.set_remote(name, url)` method that replaces (rather than appends) the catalog's git remote. This is the supported way to configure a remote: it makes the contract discoverable instead of relying on users to know that bare `git remote add` is forbidden. Raw `git remote add` against the working tree remains the configuration that triggers `CatalogConfigurationError`.

A new test (`test_multi_remote_raises_configuration_error` in `python/xorq/catalog/tests/test_catalog.py`) parametrizes `push`/`pull`/`fetch`/`sync` against a catalog with two git remotes and asserts each raises `CatalogConfigurationError`. The earlier `test_push_aggregates_failures_across_remotes` covering multi-remote aggregation is removed; if multi-remote support is later re-introduced, the bare-repo + two-remote + diverged-history setup it relied on is straightforward to re-derive.

### Escape hatch

Users who have a real multi-remote use case are directed (in the error message and in the catalog docs) to open an issue or contact us. We will design the supported semantics in collaboration with a concrete workflow rather than speculatively.

There is no in-band escape hatch. Dropping to raw `git push` bypasses `assert_consistency`, does not push the `git-annex` branch unless explicitly named, and does not run `git annex copy --to=<remote>` for content mirroring — meaning the catalog's invariants do not hold and downstream `Catalog.fetch_entries` against that remote may find metadata without content. We mention this so users do not reach for it as a workaround thinking it's "almost supported."

## Alternatives considered

### Keep multi-remote with the aggregation fix from #1898

Continue accepting any number of git remotes; rely on the push-aggregation fix to surface failures.

Rejected because:
- The aggregation fix only addresses *push* failures. `pull`, `fetch`, `assert_consistency`, and annex content placement still have undefined behavior across multiple remotes.
- It implicitly promises a feature ("multi-remote works") we have not designed, tested, or run.
- Each subsequent multi-remote bug becomes an obligation to patch a surface we did not intend to support.

### Support multi-remote with documented warnings

Keep the implementation, document the sharp edges in the catalog guide, and treat each multi-remote failure as a separate triage item.

Deferred. This is what we have today minus the explicit warning. It pushes design work onto users (who must reason about the inconsistency surface themselves) and onto future maintainers (who must keep patching aggregation-style fixes without a guiding model). We can revisit this if real users show up with use cases — see the escape hatch above.

### Refuse zero remotes too

Require exactly one git remote at all times.

Rejected because the local-only case (a freshly initialized catalog before any remote is added, or a purely local workflow during development) is genuinely useful and exercised by tests. Zero remotes today is a silent no-op for `push`/`pull`/`fetch`; the supported-configuration section above commits to keeping that behavior, since the user has not asked the catalog to talk to anything.

### Keep the surface, gate it behind a flag

Allow multi-remote only when an explicit `multi_remote=True` is passed to the catalog constructor, treating it as an opt-in unsupported configuration.

Rejected because flags-for-unsupported-features tend to drift into de facto support — users find them, build on them, and the flag becomes load-bearing without ever being designed. A clear error with a "contact us" path is more honest.

### Warn on multi-remote, keep operating

Print a deprecation-style warning the first time a multi-remote catalog is opened, but allow `push`/`pull`/`fetch`/`sync` to run with the current aggregation behavior.

Rejected on the same grounds as the flag option: a warning that's easy to ignore is functionally support. It also keeps the aggregation surface alive without committing us to the rest of the multi-remote semantics enumerated in Context, which is the worst of both worlds — we'd accumulate bug reports against a configuration we tell users is "warned" rather than supported.

## Consequences

### Positive

- The catalog's public surface matches what we test and run. No more divergence between "what the code accepts" and "what we will actually support."
- `Catalog.push()` simplifies — no aggregation loop, no per-remote partitioning of push results. (The `main`/`git-annex` two-branch half-pushed-state hazard remains and still requires its own documentation, but is now contained to one remote.)
- Users who hit the limit hear from us at the first `push`/`pull`/`fetch`/`sync` call after misconfiguration, not from a confusing partial-success three pushes later.
- `Catalog.set_remote()` makes the supported configuration path discoverable; users do not have to learn by hitting the error.
- We get a forcing function: if and when a real multi-remote use case lands, we design for it deliberately rather than retrofitting.

### Negative

- Users who today rely on multi-remote (we believe there are none, but we have not surveyed) will see a hard error after upgrading. Mitigation: the error message names the remotes and links to a contact path; the local-only path (zero remotes) is preserved.
- The aggregation fix from `6b2ab300` is reverted as part of this change. If multi-remote support is later re-introduced, both the aggregation logic (~30 lines) and the bare-repo + two-remote + diverged-history test setup would need to be re-derived; neither is preserved in tree.
- "One git remote" is enforced at the catalog API boundary, not at the underlying git repository. A user who runs `git remote add` directly on the working tree and then operates on the repo with raw git can still create the inconsistent states described in Context. We accept this — the catalog is opinionated, the underlying git repo is not. The error fires on every subsequent `push`/`pull`/`fetch`/`sync` call, not just the first one, so the misconfiguration is loud and persistent rather than one-shot.

## References

- Issue [#1898](https://github.com/letsql/xorq/issues/1898) — silent push rejection
- Commit `fe3f608b` — raise on rejected push
- Commit `6b2ab300` — aggregate push failures across all remotes (to be reverted by this ADR)
- Commit `0b40bb35` — multi-remote aggregation test (removed by this ADR; replaced by `test_multi_remote_raises_configuration_error` as live coverage)
- [ADR-0003](0003-optional-git-annex-backend.md) — git-annex backend (related; unaffected by this decision since it concerns the *special* remote, not git remotes)
