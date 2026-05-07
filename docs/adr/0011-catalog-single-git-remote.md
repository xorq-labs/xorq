# ADR-0011: Catalog supports a single git remote

- **Status:** Accepted
- **Date:** 2026-05-05
- **Deciders:** Paddy Mullen

## Context

A `Catalog` is a git repository — usually with a git-annex sidecar branch — that holds versioned build artifacts. `Catalog.push()`, `Catalog.pull()`, `Catalog.fetch()`, and `Catalog.sync()` operate over `Catalog._git_remotes`, which is *every* git remote on the underlying repo with a fetch refspec (see `python/xorq/catalog/catalog.py`). Nothing in the catalog API restricts a user to one remote: running `git remote add r2 …` on the working tree silently expands the next `Catalog.push()` to push to both `origin` and `r2`.

Issue [#1898](https://github.com/letsql/xorq/issues/1898) surfaced a silent failure mode in this design: a non-fast-forward push was rejected, and `Catalog.push()` returned without raising. The fix raises `CatalogPushError` on `REJECTED`/`REMOTE_REJECTED`/`REMOTE_FAILURE`/`ERROR` flags. Writing the multi-remote variant of that fix — push to all remotes first, aggregate failures, then raise — surfaced an open-ended set of multi-remote consistency questions we have no design for:

- **Atomicity between remotes.** `push()` to N remotes is N independent operations. Any non-empty subset can succeed; aggregating the error reports *that* it happened, not how to recover. (Single-remote does not give us atomicity either — `main` and `git-annex` are pushed in two operations inside `Catalog.push` — but it shrinks the inconsistency surface from 2N operations to 2.)
- **Annex content placement.** `git-annex` tracks per-key location across remotes. With multiple git remotes plus a special remote, "where does the content live" stops being a yes/no question.
- **Pull semantics.** `Catalog.pull` calls `Remote.pull` on each remote in turn. With divergent histories on two remotes, the first pull merges and the second fails (or merges) silently with respect to the catalog's consistency invariants.
- **`assert_consistency` against which remote?** Mutating operations (`add`, `remove`, `add_alias`) wrap their work in `Catalog.synchronizing`, which pulls from every remote then pushes to every remote. With multiple remotes, "the catalog is consistent with the remote" is undefined.
- **`remote_config` is singular.** The git-annex `RemoteConfig` resolution in `_try_resolve_annex_remote` reads one config from `remote.log`. The data model already assumes a single source of truth for credentials and bucket location.

We have no users today asking for multi-remote, no design for what the operations above *should* do, and no test coverage beyond the push-failure aggregation case. The catalog API is recent — we have room to constrain it before workflows ossify around the unsupported behavior.

## Decision drivers

- The catalog API is small and recent; constrain it before users build workflows that depend on multi-remote behavior.
- Single-remote is the only configuration we test, document, or run in production.
- Failure modes that "look like they work" are worse than failure modes that raise loudly.
- We want to hear from users with this need rather than guess at requirements.

## Decision

The catalog officially supports at most one git remote — zero (local-only) or one. Configurations with two or more git remotes are unsupported, and the catalog refuses to operate on them rather than attempting best-effort multi-remote semantics.

### What "supported" means

- `Catalog.push()`, `Catalog.pull()`, `Catalog.fetch()`, and `Catalog.sync()` are defined for repos with zero or one git remote.
  - **Zero remotes** is the local-only case (used in tests and by `init=True` before a remote is wired up). These operations are no-ops in that mode and do not raise. We accept the silent no-op here — unlike the rejected-push case, the user has not configured a remote, so there is no "I asked for X and got nothing" surprise. Local-only workflows (development, tests, fixture setup) want this.
  - **One remote** is the supported configuration; behavior is unchanged from today minus the multi-remote aggregation logic.
- A repo with two or more git remotes raises `CatalogConfigurationError` the first time one of those operations is invoked, with a message that names the remotes found.
- `Catalog.clone_from`, `Catalog.add`, `Catalog.remove`, and read-only operations are unaffected — they do not depend on a remote.

### What "git remote" means here

This ADR constrains *git* remotes — entries in `repo.remotes` with a fetch refspec, as enumerated by `Catalog._git_remotes`. The git-annex *special* remote (the S3 / directory / etc. configured via `RemoteConfig` and stored in `remote.log`) is unrelated and remains singular as it always has been:

```
catalog repo  ──push/pull──>  one git remote (e.g. github.com/org/catalog.git)
     │
     └── annex content  ──get/copy──>  one annex special remote (e.g. s3://bucket)
```

### Configuring the remote

`Catalog.set_remote(name, url, force=False)` is the supported way to wire up the git remote. It refuses by default when a remote is already configured — silent replacement turns a typo in the remote name into the deletion of the existing remote with no signal. `force=True` is the explicit opt-in for the deliberate replacement case.

The CLI mirrors the API: `xorq catalog set-remote URL [--name NAME] [--force]`.

Raw `git remote add` against the working tree is not blocked, but a second remote configured that way triggers `CatalogConfigurationError` on the next `push`/`pull`/`fetch`/`sync`.

### Where the check lives

A `_validated_git_remotes` helper reads `_git_remotes` once, raises `CatalogConfigurationError` on 2+ remotes, and returns the validated tuple. `push`, `pull`, `fetch`, and `sync` call it at the top to get a stable snapshot for the duration of the operation. Putting the guard inside `synchronizing` is insufficient: the four entry points are public methods callers invoke directly, and `sync` enters `synchronizing` itself, so the guard must run before the context-manager body. See the implementation in `python/xorq/catalog/catalog.py`.

### Escape hatch

Users with a real multi-remote use case are directed to open an issue. We will design the supported semantics in collaboration with a concrete workflow rather than speculatively. There is no in-band escape hatch: dropping to raw `git push` bypasses `assert_consistency`, does not push the `git-annex` branch unless explicitly named, and does not run `git annex copy --to=<remote>` for content mirroring — meaning the catalog's invariants do not hold. We mention this so users do not reach for it as a workaround thinking it is "almost supported."

## Alternatives considered

### Keep multi-remote with the aggregation fix proposed in PR #1899

Continue accepting any number of git remotes; rely on the push-aggregation fix to surface failures.

Rejected because:
- The aggregation fix only addresses *push* failures. `pull`, `fetch`, `assert_consistency`, and annex content placement still have undefined behavior across multiple remotes.
- It implicitly promises a feature ("multi-remote works") we have not designed, tested, or run.
- Each subsequent multi-remote bug becomes an obligation to patch a surface we did not intend to support.

### Support multi-remote with documented warnings

Keep the implementation, document the sharp edges in the catalog guide, and treat each multi-remote failure as a separate triage item.

Deferred because: this is what we have today minus the explicit warning. It pushes design work onto users (who must reason about the inconsistency surface themselves) and onto future maintainers (who must keep patching aggregation-style fixes without a guiding model). Reopen if real users show up with use cases.

### Refuse zero remotes too

Require a configured git remote at all times — disallow the local-only (zero-remote) case.

Rejected because the local-only case (a freshly initialized catalog before any remote is added, or a purely local workflow during development) is genuinely useful and exercised by tests.

### Gate multi-remote behind a flag

Allow multi-remote only when an explicit `multi_remote=True` is passed to the catalog constructor, treating it as an opt-in unsupported configuration.

Rejected because flags-for-unsupported-features tend to drift into de facto support — users find them, build on them, and the flag becomes load-bearing without ever being designed.

### Warn on multi-remote, keep operating

Print a deprecation-style warning the first time a multi-remote catalog is opened, but allow `push`/`pull`/`fetch`/`sync` to run with the current aggregation behavior.

Rejected on the same grounds as the flag option: a warning that is easy to ignore is functionally support. It also keeps the aggregation surface alive without committing us to the rest of the multi-remote semantics enumerated in Context.

### `Catalog.set_remote` silently replaces

The first cut of `set_remote` deleted any existing git remote and created the new one without complaint. The thinking was that the caller obviously means to overwrite — they just called `set_remote`.

Rejected because a typo in the remote name deletes the configured remote with no signal. The cost of asking the user to pass `force=True` for the rare deliberate-replacement case is small relative to the cost of silent destruction.

## Consequences

### Positive

- The catalog's public surface matches what we test and run. No more divergence between "what the code accepts" and "what we will actually support."
- `Catalog.push()` simplifies — no aggregation loop, no per-remote partitioning of push results.
- Users who hit the limit hear from us at the first `push`/`pull`/`fetch`/`sync` call after misconfiguration, not from a confusing partial-success three pushes later.
- `Catalog.set_remote` makes the supported configuration path discoverable and refuses to silently overwrite.
- We get a forcing function: if and when a real multi-remote use case lands, we design for it deliberately rather than retrofitting.

### Negative

- Users who today rely on multi-remote (we believe there are none, but we have not surveyed) will see a hard error after upgrading. The error message names the remotes; the local-only path (zero remotes) is preserved.
- The aggregation fix is reverted. If multi-remote support is later re-introduced, both the aggregation logic (~30 lines) and the bare-repo + two-remote + diverged-history test setup would need to be re-derived; neither is preserved in tree.
- "One git remote" is enforced at the catalog API boundary, not at the underlying git repository. A user who runs `git remote add` directly and then operates on the repo with raw git can still create the inconsistent states described in Context. We accept this — the catalog is opinionated, the underlying git repo is not. The error fires on every subsequent `push`/`pull`/`fetch`/`sync` call, so the misconfiguration is loud and persistent.
- The `main`/`git-annex` two-branch half-pushed-state hazard remains and still requires its own documentation; it is now contained to one remote.

## References

- Issue [#1898](https://github.com/letsql/xorq/issues/1898) — silent push rejection
- PR [#1899](https://github.com/xorq-labs/xorq/pull/1899) — implements this ADR alongside the push-rejection fix
- [ADR-0003](0003-optional-git-annex-backend.md) — git-annex backend (related; unaffected since it concerns the *special* remote, not git remotes)
