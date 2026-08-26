# xorq — agent guidance

Each section below takes one recurring category of agent-conduct concern
— a truthful record, real evidence, explicit consent, accurate outward
representation, sensitive data, trust boundaries, minimal footprint,
single source of truth — and names the concrete xorq mechanism for it, or
states plainly that the category doesn't apply here.

## 1. Truthful, non-rewritable record of accepted work

- "Accepted" means merged to `main` or tagged as a release. Feel free to
  rewrite, squash, or reorder commits on your own branch before it's
  opened for review or merged — the documented release flow itself
  squash-merges each release PR into `main`. Do not rewrite commits that
  are already on `main`, and do not move or delete an existing `v*` tag.
- `CHANGELOG.md` is generated from commit history by `git-cliff` during
  the release flow (see `CONTRIBUTING.md`, "Release Flow"). Don't hand-edit
  already-published sections of it; add notes the same way the release
  flow does (rerun `git-cliff`, then append manual notes as instructed).
- ADRs (`docs/adr/`) are append-only once accepted: a decision that's
  revised gets a dated `## Amendment` section, and a reversed decision gets
  marked "Superseded by ADR-NNNN" — the original text is never altered
  (see `docs/adr/template.md`).

## 2. Real evidence before asserting correctness or completion

- Reproduce a bug or name the specific behavior/contract a change is
  supposed to satisfy before writing a regression test for it. Don't
  encode a guessed root cause.
- Don't write a test that only checks that source text, a config file, or
  a CLI's own help/dry-run output contains an expected string — exercise
  the actual behavior (run the pipeline/expression/build and check its
  result) or use the tool's own validation path.
- Before claiming a change works: run the relevant slice of the suite
  (`python -m pytest`, scoped with `-m <marker>` — see `pyproject.toml`'s
  `markers` list, such as `duckdb`, `postgres`, `snapshot_check`,
  `slow`) and don't rely on markers you didn't actually run to imply
  coverage of backends you didn't test. `just download-data` and
  `just up postgres` are prerequisites for the fixtures many tests depend
  on — evidence from a suite run without them (silent skips) doesn't
  establish what it looks like it establishes.
- Closing a GitHub issue or marking a PR ready is a stronger claim than
  commenting on it. If verification isn't done, say what was tried and
  what remains instead of closing/resolving.
- **Declared contracts in comments/docstrings need a marker and a check.**
  Ordinary prose using "always," "never," or "must" carries no obligation.
  A claim is only a declared contract — and only then needs a backing
  `assert` or test — when it's written as one of a closed set of inline
  tags (`# INVARIANT:`, `# GUARANTEE:`, `# CONTRACT:`, `# PRECONDITION:`,
  `# POSTCONDITION:`) — a fixed tag marking a load-bearing comment, in the
  spirit of Rust's `// SAFETY:` idiom — or a dedicated
  `Guarantees:`/`Invariants:` section in a docstring, parallel to
  the `Parameters:`/`Returns:` sections it may already use. When adding a
  new one, prefer writing it as an inline `assert` at the same site over a
  tagged comment plus a separate test — that way there's one artifact, not
  two that can drift apart (see category 8).

## 3. Explicit consent for irreversible or high-blast-radius actions

- The steps under "Release Flow" in `CONTRIBUTING.md` (bumping the
  version, tagging `v$version_number`, pushing tags, creating the GitHub
  release that triggers publishing to PyPI) are explicitly maintainer-only.
  Do not run them, or trigger the `ci-pre-release` workflow, without the
  user explicitly asking for a release in the current conversation.
- The `[tool.uv.sources]` path override used to develop against a local
  `xorq-datafusion` checkout (see "Working with xorq-datafusion") must be
  removed before a PR merges — tests run with `--no-sources`, so leaving
  it in place silently changes what gets tested against. Don't merge with
  it present, and flag it if you find it left behind.
- Deleting or overwriting entries in a git-native catalog, an Iceberg
  `warehouse_path` directory, or a git-annex-backed large-object store (see
  `docs/adr/0003-optional-git-annex-backend.md`) is data loss for anyone
  else who reads that catalog. Treat it the same as any other destructive
  filesystem operation: confirm before deleting, prefer archiving/moving
  over removal.
- Rotating, revoking, or hardcoding backend credentials (Postgres, cloud
  storage, `OPENWEATHER_API_KEY`, and the like) is out of scope for an
  agent to do unilaterally — see category 5 for how these should be
  handled instead.

## 4. Accurate representation to external/future audiences

- Commit messages follow Conventional Commits (`fix`, `feat`, `docs`,
  `style`, plus `refactor`, `build`, `chore`, `test`, `release` as seen in
  history) and are the direct input to the auto-generated changelog. Pick
  the type that matches the actual SemVer impact — a `feat` mislabeled as
  `fix` (or vice versa) corrupts the generated release notes and the
  version-bump signal, not just the commit log's readability.
- Only add `fixes #NNNN` to a commit/PR body when the change actually
  closes that issue.
- Don't post unsolicited comments on GitHub issues or PRs; don't frame a
  routine fix as a discovered failure of prior work.

## 5. Minimize exposure of sensitive or identifying data

- Backends declare their own secret keys rather than having credentials
  inlined at call sites (see the `feat(profiles)` convention already in
  history) — follow that pattern for any new backend/profile work instead
  of hardcoding a token, password, or API key.
- Don't commit real customer or production data into `examples/`, test
  fixtures, or docs. The Postgres test credentials and
  `OPENWEATHER_API_KEY` in `CONTRIBUTING.md` are local/dev-only — treat
  any credential-shaped string you encounter as something that should stay
  in an env var or profile, not in a file that gets committed.

## 6. Trust boundaries

xorq is primarily a local CLI/library plus a git-native catalog with no
multi-tenant network service or browser-facing daemon — most of this
category doesn't have a surface to apply to, and no threat model should be
invented for one that doesn't exist.

The one real network-facing surface is the Arrow Flight server
(`python/xorq/flight/server.py`). Treat any authentication, session, or
credential-handling code on that path as a genuine boundary: don't assume
a connection is trusted without checking what actually authenticates it,
and don't let a credential intended for one backend/profile reach a
different one. If this surface grows a multi-client or multi-tenant story,
that's the point to replace this section with a real trust-boundary
description — what the boundary is, what crosses it, and what gets
verified at the crossing — rather than deferring indefinitely.

## 7. Minimum necessary footprint

- Every `__init__.py` (or star-importable module) declares an explicit
  `__all__`; that's the sole source of truth for the public surface. Don't
  use the `@public` decorator in first-party code (it's vendored-ibis-only)
  and don't add names to a public surface that aren't meant to be
  supported API.
- Default to eager, module-scope imports. Move an import into function
  scope only for one of the three documented reasons — optional/extra
  dependency, heavy import cost, or breaking an import cycle — and mark it
  `# noqa: PLC0415` so the exception is explicit and reviewable, per
  `CONTRIBUTING.md`'s "Eager vs. lazy imports" section. `PLC0415` is
  enforced by ruff (see `pyproject.toml`); don't add a blanket
  per-file/per-module ignore to work around it.

## 8. Single source of truth for tracked work, docs, and comments

- Issues are tracked on GitHub, not in a separate tracker. Search existing
  issues before opening a new one, and prefer commenting on or updating an
  existing issue over duplicating it.
- Design decisions that reasonable people could disagree on go in
  `docs/adr/`, following `docs/adr/template.md`, rather than being
  re-litigated ad hoc in scattered PR descriptions or comments.
- A comment that only restates what the code next to it already expresses
  is a second, unenforced copy of the same fact — delete it rather than
  add it. Comments earn their place by saying something the code can't
  (why, not what); a comment that references external context (an issue,
  an ADR) should point at it rather than re-explain it, so the explanation
  itself still lives in exactly one place.

## Documentation conventions

Docs and any user-facing copy follow `STYLEGUIDE.md` (capitalize "Xorq" in
body text, no Latin abbreviations, second person / present tense, sentence
case headings, and so on). Some of this is enforced mechanically, but not
uniformly: `vale` covers only the `.qmd` pages under `docs/`, and skips
`docs/reference/**` and `docs/adr/**` (see `.vale.ini` and
`.github/workflows/ci-docs-lint.yml`); `codespell` runs repo-wide from
pre-commit; the `xorq-check-style` PostToolUse hook in
`.claude/settings.json` is best-effort and exits quietly when the command
isn't installed. Don't fight or bypass that tooling — fix what it
flags — and don't read a clean run over a file outside `vale`'s scope as
confirmation that the file follows `STYLEGUIDE.md`.
