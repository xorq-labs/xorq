# xorq — agent guidance

Each section below takes one recurring category of agent-conduct concern
— a truthful record, real evidence, explicit consent, accurate outward
representation, sensitive data, trust boundaries, minimal footprint,
single source of truth — and names the concrete xorq mechanism for it, or
states plainly that the category doesn't apply here. Categories 9 and 10
cover the mechanical conventions the repo's own tooling encodes.

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
- Markers are applied by path, not by decorator (see
  `python/xorq/backends/conftest.py`): tests under
  `python/xorq/backends/<name>/` get `<name>`, everything else gets
  `core`. So `-m core` is the default slice for non-backend work.
- Catalog tests are excluded from the marker run entirely
  (`--ignore=python/xorq/catalog` in `.github/workflows/ci-test.yml`) and
  run from `ci-test-catalog.yml`, split into a `git` job and an `annex`
  job by nodeid. A new catalog test needing the `git-annex` binary must
  consume the `backend_type` fixture, or otherwise carry `[annex]` in its
  nodeid, or it silently lands in the job that can't run it — see the
  comment on that split in `ci-test-catalog.yml`.
- CI also filters `-k 'not script_execution and not slow'`, so a local
  `-m <marker>` run is a superset of what CI actually runs.
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
- The local-checkout override used to develop against `xorq-datafusion`
  (see "Working with xorq-datafusion" in `CONTRIBUTING.md`) must be
  removed before a PR merges — leaving it in silently changes what gets
  tested against. What has to go is the one `xorq-datafusion` entry with
  a `path = ...` value, not the `[tool.uv.sources]` table itself — that
  table is a permanent fixture holding three `git = ...` sources. Grep
  for a `path =` entry, not for the table name, or you'll report three
  legitimate dependencies as a leftover.
- `CONTRIBUTING.md` attributes that failure to tests running with
  `--no-sources`, but the flag appears nowhere else in the repo. What CI
  actually runs is `uv sync --locked`, which fails on an unlocked path
  source. Same outcome, different mechanism — don't go looking for a
  `--no-sources` invocation to confirm it.
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

- Commit messages follow Conventional Commits and are the direct input
  to the auto-generated changelog. `CONTRIBUTING.md` documents `fix`,
  `feat`, `docs`, and `style`; history also uses `chore`, `release`,
  `ref`/`refactor`, `perf`, `ci`, `test`, and `build`.
- `git-cliff` groups commits by the verb that starts the summary, not by
  the type: `^.*: add` and `^.*: support` become "Added", `^fix`,
  `^.*: fix`, and `^test` become "Fixed", and everything else falls
  through to "Changed" (see `commit_parsers` in `pyproject.toml`). Two
  `feat` commits land in different sections depending on their verb, so
  write the verb you actually mean — that's the token the changelog
  reads.
- `filter_unconventional` is on, so a summary that doesn't parse as a
  conventional commit is dropped from the changelog entirely rather than
  misfiled. That's the failure mode worth avoiding.
- The type still carries the SemVer claim a human reads when computing
  the next version number, which is a manual step (see "Release Flow" in
  `CONTRIBUTING.md`). Pick the type that matches the actual impact.
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

- `__all__` is the sole source of truth for a module's public surface.
  Declare it on new `__init__.py` and star-importable modules, and don't
  add names to it that aren't meant to be supported API. Don't use the
  `@public` decorator in first-party code (it's vendored-ibis-only).
- Enforcement of `__all__` is opt-in per module: the `unlisted-import`
  rule (category 9) skips any module that declares no `__all__`, so
  adding one opts that module and its importers into the check. Many
  existing modules, `python/xorq/__init__.py` among them, still don't
  declare one — that's a gap to close going forward, not a pattern to
  copy, and retrofitting one onto an old module means checking its
  importers in the same change.
- Default to eager, module-scope imports. Move an import into function
  scope only for one of the three documented reasons — optional/extra
  dependency, heavy import cost, or breaking an import cycle — and mark it
  `# noqa: PLC0415` so the exception is explicit and reviewable, per
  `CONTRIBUTING.md`'s "Eager vs. lazy imports" section. "Heavy import
  cost" always means a third-party package: stdlib imports are cheap, and
  the `deferred-stdlib` rule (category 9) forbids deferring them
  anywhere.
- `PLC0415` is enforced by ruff (see `pyproject.toml`); don't add a
  blanket per-file/per-module ignore to work around it. The existing
  `python/xorq/vendor/**` exemption is the vendored-ibis carve-out, not a
  precedent for first-party code.

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

## 9. Code conventions the tooling encodes

- `xorq-check-style` is this project's Python style enforcer, not a prose
  checker. `xorq-check-style --list` prints the current rule set and is
  the source of truth for it, so this file doesn't keep a second copy.
  Read that list once before writing Python here: the rules decide which
  library you reach for, where a class is allowed to live, and what shape
  a test takes — choices you make on nearly every file.
- Nothing runs it for you. It's a declared dev dependency with
  repo-level config in `pyproject.toml`, but no CI workflow and no
  pre-commit hook invokes it. The only automatic trigger is the
  PostToolUse hook in `.claude/settings.json`, which matches `Edit` and
  `Write` — a file you author through a shell heredoc, a `sed` call, or a
  generator script is never checked. Run `xorq-check-style` yourself
  (`--diff` lints only changed lines) when you've written Python any
  other way.
- `uv.lock` is regenerated by the `uv-lock` pre-commit hook, and
  `ci-lint` fails the build when the working tree is dirty after linting.
  If you change dependencies in `pyproject.toml`, commit the regenerated
  lockfile in the same commit.
- ruff runs only over `python`, `examples`, and `docs` (see
  `.pre-commit-config.yaml` and `.github/workflows/ci-lint.yml`). A clean
  ruff run says nothing about files outside those trees.

## 10. Documentation conventions

Docs and user-facing copy follow `STYLEGUIDE.md`. `docs/LINTING.md` is the
source of truth for which checks run and how to run them locally
(`just docs-lint`) — this file doesn't restate that table. Don't fight or
bypass the tooling; fix what it flags.

Two things about its reach that neither file makes obvious:

- `vale` only styles `.qmd` files, and zeroes out its rules for
  `docs/reference/**` and `docs/adr/**` (see `.vale.ini`). A clean lint
  run over a Markdown file — this one included — is not evidence that
  the file follows `STYLEGUIDE.md`.
- It does reach Python docstrings, indirectly:
  `docs/generate_cli_reference.py` renders Click docstrings into
  `docs/api_reference/cli/`, which is not excluded, and `ci-docs-lint`
  generates those pages before running `vale`. But that workflow triggers
  only on `docs/**`, so a docstring-only change in `python/xorq/cli.py`
  fails nothing at the time — it surfaces later, on someone else's docs
  PR.
