# Xorq — agent guidance

Each section names the concrete Xorq mechanism for one recurring
category of agent-conduct concern — a truthful record, real evidence,
consent, sensitive data, trust boundaries, minimal footprint, single
source of truth — or says plainly that the category has no surface here.
Categories 9 and 10 cover conventions the repo's own tooling encodes.

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

- Reproduce a bug, or name the behavior a change is supposed to satisfy,
  before writing a regression test. Don't encode a guessed root cause.
- Don't write a test that only checks that source text, a config file,
  or a CLI's help/dry-run output contains a string — exercise the
  behavior, or use the tool's own validation path.
- Before claiming a change works, run the relevant slice of the suite
  (`python -m pytest`, scoped with `-m <marker>` — see `pyproject.toml`'s
  `markers`) and don't let markers you didn't run imply coverage you
  don't have. `just download-data` and `just up postgres` are
  prerequisites for many fixtures; a run without them skips silently, so
  its evidence doesn't establish what it looks like it establishes.
- Markers come from `python/xorq/backends/conftest.py`, applied by path
  at collection: tests under `python/xorq/backends/<name>/` get
  `<name>`, everything else gets `core`. That conftest only registers
  once collection reaches `python/xorq/backends`, so narrowing the path
  *and* passing `-m core` deselects everything —
  `pytest python/xorq/tests -m core` collects 0 of 855. Scope by path or
  by marker over the whole tree, not both; a zero-collection run is the
  failure mode to watch for.
- Catalog tests are excluded from the marker run entirely
  (`--ignore=python/xorq/catalog` in `.github/workflows/ci-test.yml`) and
  run from `ci-test-catalog.yml`, split into a `git` job and an `annex`
  job by nodeid. A new catalog test needing the `git-annex` binary must
  consume the `backend_type` fixture, or otherwise carry `[annex]` in its
  nodeid, or it silently lands in the job that can't run it — see the
  comment on that split in `ci-test-catalog.yml`.
- `ci-test.yml` filters `-k 'not script_execution and not slow'`, but
  excluded-here means run-elsewhere: `ci-test-slow.yml` runs the `slow`
  levels, and `ci-test-examples.yml` runs `test_examples.py` with no
  `-k`. There are 12 `ci-test-*.yml` workflows; the ones named here are
  not the whole of CI.
- Closing a GitHub issue or marking a PR ready is a stronger claim than
  commenting on it. If verification isn't done, say what was tried and
  what remains instead of closing/resolving.

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
  `--no-sources`, but that flag appears nowhere else in the repo. What
  CI actually runs is `uv sync --locked` (category 9). Same outcome.
- Confirm before deleting state that someone else reads: a git-native
  catalog others pull from, or a git-annex-backed large-object store
  (see `docs/adr/0003-optional-git-annex-backend.md`). Losing those
  costs other people work, so prefer archiving or moving over removal.
  Local scratch isn't in that class — `warehouse` is gitignored, and the
  pyiceberg fixtures mint and discard `warehouse_path` directories under
  `tmp_path` constantly. The test is whether anyone else reads it.
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
  the type (see `commit_parsers` in `pyproject.toml`): `add` and
  `support` become "Added", `remove` and `delete` become "Removed",
  `^fix`, `^test`, and `: fix` become "Fixed", and everything else falls
  through to "Changed". The parsers are ordered and first-match-wins, so
  `fix(lint): remove ...` lands in Removed, not Fixed. Write the verb
  you actually mean — that's the token the changelog reads.
- `filter_unconventional` is on, so a summary that doesn't parse as a
  conventional commit is dropped from the changelog entirely rather than
  misfiled. That's the failure mode worth avoiding.
- The type still carries the SemVer claim a human reads when computing
  the next version number by hand (see "Release Flow" in
  `CONTRIBUTING.md`). Pick the type that matches the actual impact.
- Only add `fixes #NNNN` to a commit/PR body when the change actually
  closes that issue.
- Don't post unsolicited comments on GitHub issues or PRs; don't frame a
  routine fix as a discovered failure of prior work.

## 5. Minimize exposure of sensitive or identifying data

- Backends declare their own secret keys rather than having credentials
  inlined at call sites — `Backend._secret_keys` (see
  `python/xorq/backends/postgres/__init__.py:40`), mirrored by
  `con_name_to_secret_keys` in
  `python/xorq/vendor/ibis/backends/profiles.py`. Follow that pattern
  for new backend or profile work instead of hardcoding a token,
  password, or API key.
- Don't commit real customer or production data into `examples/`, test
  fixtures, or docs. The Postgres test credentials and
  `OPENWEATHER_API_KEY` in `CONTRIBUTING.md` are local/dev-only — treat
  any credential-shaped string you encounter as something that should stay
  in an env var or profile, not in a file that gets committed.

## 6. Trust boundaries

Xorq is primarily a local CLI/library plus a git-native catalog with no
multi-tenant network service or browser-facing daemon — most of this
category doesn't have a surface to apply to, and no threat model should be
invented for one that doesn't exist.

The one real network-facing surface is the Arrow Flight server
(`python/xorq/flight/server.py`). Treat any authentication, session, or
credential-handling code on that path as a genuine boundary: don't
assume a connection is trusted without checking what authenticates it,
and don't let a credential intended for one backend or profile reach a
different one.

## 7. Minimum necessary footprint

- `__all__` is the sole source of truth for a module's public surface.
  Declare it on new `__init__.py` and star-importable modules, and don't
  add names to it that aren't meant to be supported API. Don't use the
  `@public` decorator in first-party code (it's vendored-ibis-only).
- Nothing enforces that. The `unlisted-import` rule (category 9) never
  fires here: it resolves imported modules under `src-roots`, which
  defaults to `("src", ".")` while this repo's source root is `python/`,
  and `pyproject.toml` configures only `[tool.xorq-style.print]`. Adding
  `src-roots = ["python"]` turns it on — and even then a non-literal
  `__all__` such as `python/xorq/ml.py`'s `[*ml.__all__]` stays
  invisible, because the rule reads only statically-evaluable lists. A
  clean style run is not evidence you got `__all__` right.
- `python/xorq/vendor/` is vendored ibis, regenerated by the `vendoring`
  tool (`[tool.vendoring]` in `pyproject.toml`). Edits there get
  overwritten: `vendoring sync` rewrites the tree in place, deleting
  first-party files added inside it, then errors out on an import it
  can't rewrite and tells you to add a patch — but `patches-dir` points
  at `tasks/patches`, and `tasks/` doesn't exist in the checkout, so
  there is no working way to express one. Change first-party code
  instead, and don't run `vendoring sync` casually.
- Default to eager, module-scope imports. Move an import into function
  scope only for one of the three documented reasons — optional/extra
  dependency, heavy import cost, or breaking an import cycle — and mark
  it `# noqa: PLC0415`, per `CONTRIBUTING.md`'s "Eager vs. lazy
  imports". "Heavy" always means a third-party package; the
  `deferred-stdlib` rule (category 9) forbids deferring stdlib anywhere.
- `PLC0415` is enforced by ruff; don't add a blanket per-file ignore to
  work around it. The `python/xorq/vendor/**` exemption is the
  vendored-ibis carve-out, not a precedent for first-party code.

## 8. Single source of truth for tracked work, docs, and comments

- Issues are tracked on GitHub, not in a separate tracker. Search existing
  issues before opening a new one, and prefer commenting on or updating an
  existing issue over duplicating it.
- Design decisions that reasonable people could disagree on go in
  `docs/adr/`, following `docs/adr/template.md`, rather than being
  re-litigated ad hoc in scattered PR descriptions or comments.
- A comment that only restates the code next to it is a second,
  unenforced copy of the same fact — delete it rather than add it. A
  comment referencing external context (an issue, an ADR) should point
  at it rather than re-explain it.

## 9. Code conventions the tooling encodes

- `xorq-check-style` is this project's Python style enforcer, not a
  prose checker. `xorq-check-style --list` prints the rule set and is
  the source of truth for it. Read that list once before writing Python
  here: the rules decide which library you reach for, where a class may
  live, and what shape a test takes.
- Nothing runs it for you. It's a declared dev dependency with
  repo-level config in `pyproject.toml`, but no CI workflow and no
  pre-commit hook invokes it. The only automatic trigger is the
  PostToolUse hook in `.claude/settings.json`, which matches `Edit` and
  `Write` — a file you author through a shell heredoc, a `sed` call, or
  a generator script is never checked, so run it yourself in that case.
- Use `git diff | xorq-check-style --diff` to answer "did I break
  style?". Whole-file output is dominated by what was already there: the
  repo carries roughly 7,300 violations, and one `Edit` to
  `python/xorq/cli.py` makes the hook report 50 of them spanning the
  whole file. Pre-existing violations are not yours to fix.
- `uv.lock` is regenerated by the `uv-lock` pre-commit hook. The gate on
  a stale lockfile isn't the lint step — `ci-lint.yml` runs `ruff check`
  without `--fix`, so it can't dirty the tree — it's `uv sync --locked`,
  in both `ci-lint.yml` and `ci-test.yml`, which exits 1 outright. If
  you change dependencies in `pyproject.toml`, commit the regenerated
  lockfile in the same commit.
- CI runs ruff only over `python`, `examples`, `docs`, and `scripts`
  (`ci-lint.yml`). Pre-commit reaches wider: those four are hook
  arguments, and pre-commit appends the changed filenames on top, while
  `ruff-format` carries no path restriction at all. A clean CI ruff run
  says nothing about files outside those four trees.

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
