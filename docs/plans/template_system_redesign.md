# Template System Redesign

Status: plan finalized after design review. Ready to implement on a fresh branch off `main`.

## Problem

`xorq init` downloads a template repo (`xorq-labs/xorq-template-{name}`) at a pinned commit SHA. Each template ships its own `pyproject.toml`, `uv.lock`, and `requirements.txt`. The templates also carry `[tool.uv.sources].xorq = { git = "https://github.com/xorq-labs/xorq" }`, which makes uv resolve `xorq` from HEAD of main — so the version range in `dependencies` is decorative and the shipped `uv.lock` is stale within hours of being written.

Today the only way to keep them in sync is the "arduous update cycle":

1. Cut a new `xorq` release.
2. Manually open each `xorq-template-*` repo, bump its xorq dep, re-export `uv.lock` and `requirements.txt`, commit.
3. Bump the three commit SHAs in `python/xorq/init_templates.py`.
4. Cut another `xorq` release.

Consequences:

- Bugs in the template ↔ xorq integration (e.g. the `xorq uv build` failures that drove `00e62199`, `0feaf6c7`, `647cd3e1`) are not caught until after release.
- `test_init_uv_build_uv_run` papers over the issue by `rm requirements.txt` after init.
- Tests can't exercise "current tree of xorq against current templates" — they always run the pinned, stale combination.

An abandoned fix exists at commit `d5de1380` ("initial pass at rewriting the templates to latest version of xorq on init") — useful reference, ~40% of the final design.

## Goals

1. After `xorq init`, the generated project's `xorq` dependency matches the `xorq` that ran `init` — whether that was a PyPI install, an editable dev install, or a git install.
2. The `xorq` test suite runs templates against the **current working tree** of xorq, so template breakage is caught in CI before release.
3. Releasing xorq no longer requires bumping template repos and re-pinning SHAs.

## Non-goals

- Inlining the template files into this repo. Templates stay in `xorq-labs/xorq-template-*`.
- Replacing `uv` / introducing a different package manager.
- Reworking which templates exist or what they contain.
- Preserving the ability to `git clone + uv sync` a template repo directly. By design, templates become usable only through `xorq init`. Template contributors iterate via `xorq init --xorq-spec '…' --template <name> --branch <pr-branch>`, which exercises the same path users hit.

## Design

### 1. Placeholder in template `pyproject.toml`

Each template's `pyproject.toml`:

```toml
[project]
dependencies = [
    "xorq[duckdb] @ LATEST",
    # other deps...
]
# NO [tool.uv.sources] block — single source of truth
```

`LATEST` is literal. It is valid PEP 508 syntax (URL form), so `tomlkit` round-trips it cleanly, but `uv lock` rejects it as an unresolvable URL — which is the loud "you bypassed `xorq init`" signal we want. Extras (e.g. `[duckdb]`) are preserved by the rewrite.

Templates ship **no `uv.lock`** and **no `requirements.txt`**. (See the addendum at the end of this doc for why this is conventional, not a "subverted lockfile.")

### 2. Spec resolution

`resolve_xorq_spec(override=None)` in `python/xorq/init_templates.py` returns the full PEP 508 spec to substitute. Detection by `importlib.metadata.distribution("xorq").read_text("direct_url.json")`:

| Install mode | `direct_url.json` shape | Substituted spec |
|---|---|---|
| PyPI release | absent | `xorq[extras] == {xorq.__version__}` |
| Editable / non-editable local dir | `dir_info` (with or without `editable: true`); `url: file://…` | `xorq[extras] @ {url}` |
| Wheel/sdist archive install | `archive_info` | `xorq[extras] @ {url}` |
| git install | `vcs_info` | `xorq[extras] @ git+{url}@{commit_id}` |
| Unrecognized shape | none of the above | hard error: print the contents, instruct user to pass `--xorq-spec` |
| Override (any of the above) | `--xorq-spec` CLI flag | the override value verbatim |

The override is a full PEP 508 spec (e.g. `xorq[duckdb] == 0.3.25` or `xorq[duckdb] @ git+https://github.com/xorq-labs/xorq@main`). If the user's spec extras don't match the template's extras, warn but proceed — the user opted in.

In CI, xorq is installed editable via `uv sync`. Auto-detection produces `xorq @ file:///github/workspace`, so `test_init_uv_build_uv_run` exercises the template against the in-tree xorq with no extra plumbing. Template ↔ xorq drift fails this test before release.

### 3. Post-download steps in `init_command`

```python
path = download_unpacked_xorq_template(path, template, branch=branch)
if has_latest_placeholder(path):
    spec = resolve_xorq_spec(override=xorq_spec)
    rewrite_template_xorq_dep(path, spec)  # mutates pyproject.toml, deletes uv.lock + requirements.txt
    if not no_lock:
        run_uv_lock(path)                  # `uv lock` in $path
else:
    warn(
        "template does not contain `xorq @ LATEST`; skipping substitution and lock. "
        "Update template to the new placeholder format."
    )
print(f"initialized xorq template `{template}` to {path} (xorq pinned to `{spec}`)")
```

`rewrite_template_xorq_dep`:
- Loads `pyproject.toml` with `tomlkit` (preserves comments/formatting; already a runtime dep).
- Walks `[project].dependencies`. Matches the entry `xorq @ LATEST` or `xorq[extras] @ LATEST` (regex: `^xorq(\[[^\]]+\])?\s*@\s*LATEST$`). Replaces it with the resolved spec.
- Strips `[tool.uv.sources].xorq` if present (only relevant for templates not fully converted yet).
- Deletes `uv.lock` and `requirements.txt`.
- Errors loudly if no matching dep is found.

`run_uv_lock`:
- `subprocess.run(["uv", "lock"], cwd=path, check=False)`.
- On failure: **leave the substituted `pyproject.toml` in place**, do not delete the target directory, print the uv error verbatim, exit non-zero. ("During development we want failures to scream; a dev needs to look at the half-done state.")
- Skippable via `--no-lock`.

### 4. CLI surface

`xorq init` gains:

- `--xorq-spec SPEC` — explicit override of the auto-detected spec. Full PEP 508 form (`"xorq[duckdb] == 0.3.25"`, `"xorq @ git+https://github.com/xorq-labs/xorq@abc123"`).
- `--no-lock` — skip the `uv lock` step (user runs it themselves).

Existing flags (`--path`, `--template`, `--branch`) unchanged.

### 5. Template repo workflow (post-conversion)

Each `xorq-template-*` repo:

- `pyproject.toml`: `xorq[extras] @ LATEST` in `[project].dependencies`. No `[tool.uv.sources].xorq`.
- No `uv.lock`. No `requirements.txt`.
- README updated: the repo is only usable via `xorq init`. Contributors test changes by running `xorq init --xorq-spec '…' --template <name> --branch <pr-branch>`.
- Template repo CI (separate work, not in scope here): could `pip install xorq=={latest}` and then run `xorq init` against itself to verify the template builds.

### 6. Test strategy

Update `python/xorq/tests/test_cli.py`:

- `test_init_command_default` / `test_init_command_sklearn`: after init, assert the generated `pyproject.toml` contains no `LATEST`, contains `xorq` pinned to *some* concrete spec, and that `uv.lock` exists (because init runs `uv lock`).
- `test_init_uv_build_uv_run` (slow): drop the `rm requirements.txt` workaround. With editable auto-detect, the template builds against the in-tree xorq. This becomes the **CI gate for template ↔ xorq drift**.
- New `test_resolve_xorq_spec`: unit-level coverage for each detection branch (mock `importlib.metadata.distribution`).
- New `test_rewrite_template_xorq_dep`: fixture `pyproject.toml`, run rewrite, assert deps/sources/lockfile state.
- New `test_init_old_template_fallback`: confirm that running init against an old-format template (no `LATEST`) emits a warning and falls back gracefully.

### 7. Rollout

Three steps, independent PRs:

1. **xorq PR (this one)**: ship `resolve_xorq_spec`, `rewrite_template_xorq_dep`, `run_uv_lock`, `--xorq-spec` / `--no-lock` flags, updated tests. Includes back-compat fallback so old templates (no `LATEST`) keep working with a deprecation warning. No SHA changes yet. End users see no change.
2. **Three template-repo PRs**, one per template: switch to `xorq[extras] @ LATEST`, delete `[tool.uv.sources].xorq`, delete `uv.lock`, delete `requirements.txt`, update README.
3. **Back here**: bump SHAs in `python/xorq/init_templates.py` to the new template commits, on each merge.

SHA pinning stays in `init_templates.py` indefinitely. The bump frequency drops sharply because template files rarely change once converted (xorq version drift is no longer a reason to touch them).

## Files to change (xorq PR)

- `python/xorq/init_templates.py` — add `resolve_xorq_spec`, `rewrite_template_xorq_dep`, `run_uv_lock`, `has_latest_placeholder`.
- `python/xorq/cli.py` — `init` command gains `--xorq-spec` and `--no-lock`; `init_command` orchestrates the new steps.
- `python/xorq/tests/test_cli.py` — assertions per §6.
- `python/xorq/common/utils/download_utils.py` — unchanged.

`tomlkit>=0.12` is already a runtime dep (`pyproject.toml`); no new deps.

---

## Addendum: how other package managers handle the lockfile-on-scaffold question

Raised during design review: "we're really subverting the meaning of `uv.lock` here." Worth pinning down what the convention actually is across the ecosystem before deciding whether init should auto-run `uv lock`.

### Empty-scaffold commands: none ship a lockfile

| Tool | Command | Lockfile shipped? | Verified by |
|---|---|---|---|
| uv (Python) | `uv init` | No (`pyproject.toml`, `main.py`, `.python-version`, `README.md`, `.gitignore`) | Local run |
| Cargo (Rust) | `cargo init` | No (`Cargo.toml`, `src/main.rs`, `.gitignore`) | Local run |
| pnpm (JS) | `pnpm init` | No (`package.json` only) | [pnpm docs](https://pnpm.io/cli/init) |
| npm (JS) | `npm init` | No (`package.json` only) | [npm docs](https://docs.npmjs.com/cli/v10/commands/npm-init) |
| Poetry (Python) | `poetry new` | No (`pyproject.toml`, `README.md`, source + tests dirs) | [Poetry docs](https://python-poetry.org/docs/cli/#new) |

Universal convention: the lockfile is generated by the user's first dependency-resolving command (`uv sync` / `cargo build` / `npm install` / `poetry install`), not by the scaffolder. The lock is a *user-project* artifact, not a *scaffold* artifact.

### Template scaffolders: mixed, but special-case templates

The closer comparison to `xorq init` is template-based scaffolders that pull a configured project, not empty inits:

- **cookiecutter** (Python): templates typically ship `pyproject.toml`/`setup.py` but not a lockfile. Author can include one if they want; convention says no.
- **cargo generate** (Rust): depends on the template author. Templates that represent finished applications sometimes ship `Cargo.lock`; skeleton templates usually don't.

So if you squint: a fully-built, version-pinned template could justify shipping a lock. *But* the xorq case has a property those don't:

### Why xorq templates are a special case

The `xorq` dep's version is **literally not knowable at template-build time** — it depends on which `xorq` CLI the user runs `xorq init` with. A normal cookiecutter template could lock `pandas == 2.1.0`, `sklearn == 1.4.0`, etc. and ship a working lock. The xorq template cannot, because there's no fixed value for `xorq` until the user invokes init.

This is why the current templates' shipped `uv.lock` is **not** a meaningful reproducibility artifact: the templates use `[tool.uv.sources].xorq = { git = "https://github.com/xorq-labs/xorq" }`, which resolves to HEAD of main, which moves continuously. The shipped lock pins a specific `xorq` commit, but that pin is stale within hours of being written. The lockfile's contract — "this graph reproduces" — is already broken before the user sees it.

### What this implies for the design

Two coherent positions, both defensible:

1. **Follow the universal convention.** Don't ship a lock; let the user generate it at first resolve. `xorq init` writes `pyproject.toml`, exits, and the user runs `uv sync` (or `uv lock`). This is what every other init/new command does. Clean ownership: the lock is unambiguously a user artifact.

2. **Generate the lock at init time as a courtesy.** Slightly *more* than the convention does, but justifiable because (a) `xorq init` is a heavier action than `uv init` — it pulls a configured project, not a blank one, so a one-extra-step delta is reasonable; (b) running `uv lock` immediately surfaces resolution failures (e.g. broken xorq spec) at init time rather than at `xorq uv build` time, which is when the user is least expecting them; (c) downstream tooling (`xorq uv build`) hard-errors without a lock, so the user *will* have to run `uv lock` before doing anything useful.

Neither position "subverts" `uv.lock`. Position 1 treats it exactly like every other ecosystem treats locks. Position 2 treats init as the user's first resolve, on their behalf — semantically identical to the user running `uv lock` themselves, just bundled into one command.

The position we should **not** take is the current one: shipping a lock with the template that was never reproducible to begin with. That *does* subvert the lockfile's contract.

**Resolved**: Position 2 (init runs `uv lock`), skippable with `--no-lock`.
