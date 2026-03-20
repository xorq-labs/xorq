# ADR-0001: Use git-annex over git-lfs for catalog artifact storage

- **Status:** Accepted
- **Date:** 2026-03-18
- **Context area:** `python/xorq/catalog/`

## Context

The xorq catalog needs to track large artifacts (cached query results, serialized expressions, etc.) alongside code in git repositories. Users clone catalog repos to discover and compose data assets, but should **not** be forced to download all artifact content on clone — only metadata.

Two mature tools exist for managing large files in git: **git-lfs** and **git-annex**. Both replace large file content with lightweight pointers in the git tree, but they differ in defaults, storage model, and operational surface area.

## Decision

Use **git-annex** for managing large artifacts in catalog repositories.

## Rationale

### Primary driver: default-no-download semantics

git-annex replaces large files with symlinks into `.git/annex/`. On clone, only pointer metadata is transferred — content must be explicitly fetched with `git annex get <path>`. This makes the safe, lightweight clone the default behavior with no user configuration required.

git-lfs achieves similar behavior only via `git lfs install --skip-smudge` followed by selective `git lfs pull --include=...`. This is opt-in configuration rather than a structural default. For a catalog that many consumers will clone, "safe by default" matters more than "safe if configured correctly."

### Secondary driver: fine-grained content control

- `git annex get <path>` / `git annex drop <path>` provide per-file, per-operation control over what is present locally.
- git-annex tracks content availability across multiple remotes, enabling queries like "which remote has this file?" without downloading it.
- Special remotes (S3, directory, rsync, etc.) allow heterogeneous storage backends behind a single interface — useful for supporting both local directory caches and cloud storage.

### Alternatives considered

**git-lfs** was the primary alternative. It would be simpler operationally and has first-class support on GitHub/GitLab. It was rejected because:

- Its default behavior (download everything on checkout) is the opposite of what we want.
- `--skip-smudge` is a per-clone configuration flag, not a structural property of the repository. This is error-prone for a tool that many users will clone.
- Include/exclude patterns are global config, not per-operation — less flexible for selective materialization.
- Single LFS endpoint per remote limits storage tiering.

**No large-file tool** (raw git or out-of-band storage with URL references) was rejected because we want content-addressed integrity, deduplication, and a unified `git clone` workflow.

## Consequences

### Positive

- Clone is lightweight by default — consumers get metadata only.
- Per-file `get`/`drop` enables selective materialization of catalog artifacts.
- Multiple special remotes allow flexible storage backends (local directory for dev, S3 for production).
- Content-addressed storage provides integrity checking and deduplication.

### Negative

- **Operational complexity** — git-annex has a large surface area and a steep learning curve. Contributors familiar with git-lfs will need onboarding.
- **CI configuration burden** — annex operations require git identity configuration and explicit init. Failures must be handled (see hardening in `c61b2a94`, `25b858fd`).
- **Symlink model** — annex symlinks can break tools that don't follow them (some editors, `COPY` in Dockerfiles, Windows without WSL).
- **Windows support** — git-annex on Windows is fragile. If Windows support becomes a requirement, this decision should be revisited.
- **Platform integration** — GitHub/GitLab have no native annex support. We manage our own special remotes and cannot rely on platform-provided storage or UI previews.
- **Maintenance risk** — git-annex is primarily maintained by a single developer (Joey Hess). git-lfs is backed by GitHub/Microsoft.

### Mitigations

- Keep the annex layer tightly encapsulated behind `python/xorq/catalog/annex.py` so that annex complexity does not leak into the user-facing catalog API.
- Pin a minimum annex version and test against it in CI.
- If annex operational costs become disproportionate, re-evaluate git-lfs with a `--skip-smudge`-by-default wrapper.
