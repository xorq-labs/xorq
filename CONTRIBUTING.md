# Contributing Guide

## Setting up a development environment

 > [!NOTE]
  > **macOS users:** Some dependencies require `cmake` and `libomp`. Install them
  via Homebrew with:
  > `brew install cmake libomp`

This assumes you have uv installed, otherwise please follow these [instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# fetch this repo
git clone git@github.com:xorq-labs/xorq.git
# set uv run command to not sync
export UV_NO_SYNC=1
```

```bash
# prepare development environment and install dependencies (including current package)
uv sync --all-extras --all-groups
# activate the venv
source .venv/bin/activate
# set up the git hook scripts
uv run pre-commit install
# ignore bulk-reformat commits in git blame (per clone; git never accepts this from the repo itself)
git config --replace-all blame.ignoreRevsFile .git-blame-ignore-revs
```
> [!IMPORTANT]
> Rename `.gitignore.template` to `.gitignore` 

### git blame and bulk reformats

`.git-blame-ignore-revs` lists the commits `git blame` should skip. Git
deliberately never accepts `blame.ignoreRevsFile` from the repository itself,
hence the `git config` line above. `pre-commit install` also installs a
`post-checkout` hook that sets it -- not the install itself, the next checkout
after it -- so clones set up before that hook existed need `uv run pre-commit
install` once more. GitHub's web blame honors the file with no setup at all.

One caveat: with the setting on, `git blame` exits 128 with `fatal: could not
open object name list: .git-blame-ignore-revs` at any rev predating the file
(2024-05-17), so bisecting into early history or checking out a tag up to
`v0.1.2.post` breaks blame entirely. There is no per-command override --
`blame.ignoreRevsFile` is multi-valued, so `-c blame.ignoreRevsFile=` and
`--ignore-revs-file=""` append to the list rather than replace it. Unset it for
the duration instead, and the next checkout puts it back:

```bash
git config --unset-all blame.ignoreRevsFile
```

## Dev container

Development tooling (devcontainer, worktree helpers) lives in a separate repository:

```bash
git clone git@github.com:xorq-labs/devcontainer.git ../devcontainer
```

Add the tooling to your PATH in `.envrcs/.envrc.user`:

```bash
PATH_add $direnv_root/../devcontainer/dev
```

See the [devcontainer repo](https://github.com/xorq-labs/devcontainer) for full usage details.

## Running the test suite
Install the [just](https://github.com/casey/just#installation) command runner, if needed.
Download example data to run the tests successfully.

```bash
just download-data
```

Populate the environment variables:

```bash
export POSTGRES_DATABASE=ibis_testing
export POSTGRES_HOST=localhost
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_PORT=5432
```

> [!NOTE]
> Some tests (e.g., `test_script_execution[weather_flight]`) require additional environment variables like `OPENWEATHER_API_KEY`.
> See the [Environment Variables documentation](docs/api_reference/backends/env_variables.qmd#weather-api-configuration-envweathertemplate) for details on obtaining and configuring these variables.

To test the code:
```bash
# make sure you activate the venv using "source venv/bin/activate" first
just up postgres # some of the tests use postgres
python -m pytest # or pytest
```

## Style checks

Ruff and `xorq-check-style` run on every pull request. `xorq-check-style`
enforces the conventions below — import placement, `__all__`, `pytest` idioms —
that Ruff has no rule for. `pre-commit install` runs it over what you staged, on
any commit that stages Python; the command below reproduces what CI will say
about the branch.

It runs as two gates, both in `.github/workflows/ci-lint.yml`:

- **Changed lines**, on pull requests and as the pre-commit hook. Every rule
  applies except those in `disable=` in `scripts/check-style-diff.sh`, which CI
  and the hook both run so they cannot drift. Pre-existing violations in a file
  you edit do not block you.
- **Whole repo**, on every build. Only the rules already at zero everywhere —
  whatever is left after the `--disable` list. Drive a rule's count to zero,
  move it off that list, and it can never come back.

Before either gate runs, CI runs `scripts/style_tests/`. `xorq-check-style` exits
0 both on a clean file and on one it never examined, so every rule owns a file it
is required to flag; a rule that stops firing fails there rather than reporting a
comfortable zero. The suite also reads the two `--disable` lists and the three
copies of the linted-tree list rather than restating any of them, so a typo, a
rule that leaves the checker, or a tree list that agrees in only two of its three
files fails a test instead of silently enforcing a set nobody chose.

To reproduce what CI will say about your branch, run the gate over the range CI
uses — `origin/$GITHUB_BASE_REF...HEAD`, which for a pull request against main
is:

```bash
uv run scripts/check-style-diff.sh origin/main...HEAD
```

The pre-commit hook runs the same script over your staged changes. Run it by
hand with `uv run pre-commit run xorq-check-style` — without `--all-files`,
which turns off the stash that makes the working tree match the index.

To see every rule a single file breaks, including the ones the gate lets
through, check the file directly:

```bash
uv run xorq-check-style python/xorq/expr/api.py
```

Suppress a single line with a trailing pragma:

```python
if ctx._protected_args:  # xorq-style: disable=protected-access
```

The pragma binds to the line it sits on, so a `ruff format` re-wrap can strand
it on the wrong one. The changed-lines gate catches that, because the re-wrap
puts the line back in the diff.

## Module and import conventions

These rules keep the public API explicit and import time predictable across the
package. They apply to first-party code under `python/xorq/`; vendored code
under `python/xorq/vendor/` follows upstream conventions and is exempt.

### Declaring the public API with `__all__`

- Every `__init__.py` (and any module meant to be imported with `*`) declares
  an explicit `__all__` listing the names it exports. This is the single source
  of truth for the public surface and what IDEs and doc tooling read.
- Do **not** use the `@public` decorator in first-party code. It mutates
  `__all__` at runtime, which hides the public surface from static tools and
  forces `# noqa: PLE0604`. `@public` remains only in vendored ibis.
- Names absent from `__all__` are private: they can still be imported by their
  fully qualified path, but they are excluded from `from module import *` and
  are not part of the supported API.

### Eager vs. lazy imports

Top-level (eager, module-scope) imports are the default. Move an import into
function scope (a "lazy" import) only for one of these reasons:

- **Optional dependencies** — a backend or feature whose third-party package is
  not a hard install requirement (e.g. `gcsfs`, `snowflake`, `pyiceberg`).
  Deferring the import keeps `import xorq` working when the extra is absent and
  surfaces the missing dependency only when the feature is actually used.
- **Heavy imports** — packages with a large import cost that most code paths do
  not touch, kept out of the base import time.
- **Breaking an import cycle** — when a module-level import would create a
  circular dependency.

Core xorq modules and cheap standard-library imports load eagerly at module
scope. Ruff enforces this split via `PLC0415` (import-outside-top-level): every
deliberate lazy import must carry a `# noqa: PLC0415` comment, which makes each
one an explicit, reviewable decision rather than an accident.

## Commenting

**Justify a rule with a ratio, not a count.** A comment that explains a threshold
by what was measured is worth more than one that just states it — without the
evidence the next reader takes the rule for a preference and relaxes it. But
exact numbers rot, and duplicated exact numbers rot faster: `adr_check.py`
shipped with "24 of the 119" in a test docstring beside "25 of the 106" in the
module it tested. State the ratio at the rule it justifies, once. Never in a test
docstring, where the assertion is already the specification and a count is a
claim nothing checks. If a number has to be exact, compute it rather than write
it down.

**Don't claim parity in prose.** A comment saying one thing "mirrors" another
asserts a property nothing enforces, and the two drift while the comment goes on
insisting they haven't: the pre-commit style hook claimed to mirror the CI gate
in the same commit that gave it neither the gate's `core.quotepath=false` nor
its `pipefail`. This holds for configuration as much as for code — a path list
or a pinned version copied into a second file rots the way a count does. Make
them one thing both callers run; failing that, write the test that reads both
and compares, and have the comment point at the test rather than assert the
agreement. Where neither is possible, say what differs instead of claiming
nothing does.

## Writing the commit

xorq follows the [Conventional Commits](https://www.conventionalcommits.org/) structure.
In brief, the commit summary should look like:

    fix(types): make all floats doubles

The type (e.g. `fix`) can be:

- `fix`: A bug fix. Correlates with PATCH in SemVer
- `feat`: A new feature. Correlates with MINOR in SemVer
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)

If the commit fixes a GitHub issue, add something like this to the bottom of the description:

    fixes #4242

## Recording an architecture decision

Choices where reasonable people could disagree, and where the reasoning outlives
the diff, belong in an architecture decision record under `docs/adr/`. Start one
with:

```sh
python3 scripts/adr_new.py my-decision
```

An ADR takes the number of the pull request that adds it, so it is written as
`XXXX-my-decision.md` and numbered by `python3 scripts/adr_rename.py` once the
pull request exists. Cite an ADR that has no number yet by its slug
(`ADR-my-decision`); both forms keep resolving.

See [docs/adr/README.md](docs/adr/README.md) for the full convention. The
`ci-adr` workflow enforces it: its `adr` and `adr-tests` jobs are required
status checks on `main`.

## Working with the documentation

> [!TIP]
> Read our [style guide](https://github.com/xorq-labs/xorq/blob/main/STYLEGUIDE.md) to 
> learn more about our writing style.

To build or preview the documentation locally, follow the steps below.

1. **Install Quarto**

   Follow the official Quarto installation guide:
   [https://quarto.org/docs/get-started/](https://quarto.org/docs/get-started/)


2. **Install required Python dependencies** 

   Run the following to install all development, test, and documentation dependencies:

   ```bash
   uv sync --locked --group dev --group test --group docs --extra examples
   ```

3. **Build and preview the documentation**

   ```bash
   cd docs  # ensure you are in the docs directory
   uv run --no-sync quartodoc build --verbose --config _quarto.yml
   uv run --no-sync quarto preview
   ```

This will build the API reference and launch a local preview server so you can iterate on documentation changes.


## Working with xorq-datafusion

xorq-datafusion is a Backend developed by xorq labs based on the DataFusion query engine, occasionally we make change to it
that must be reflected in xorq. So when working with both repos, the following workflow is proposed:

1. Add the following config in `pyproject.toml` of `xorq` (see more in [here](https://docs.astral.sh/uv/concepts/projects/dependencies/#path)):

```toml
[tool.uv.sources]
xorq-datafusion = { path = "local/path/to/xorq-datafusion-repo" }
```

2. Make changes to `xorq-datafusion` (add tests if required).
3. Verify the change in xorq via tests.
4. Commit the changes to `xorq-datafusion`, open a PR, commit to main, and release a new version of `xorq-datafusion`.
5. Remove `tool.uv.sources` from the `pyproject.toml` and open a PR with the respective change to `xorq`.

Notice that our test run with `--no-sources` so they will fail if a new version of `xorq-datafusion` with the required 
change is not present in PyPI. 

## Release Flow
***This section is intended for xorq maintainers***

On main, run `scripts/release.sh [--no-pause] <major|minor|patch>` to do steps 1-6. It refuses to start on a dirty tree, an `origin` other than xorq-labs/xorq, or a main with unpushed commits, and it pulls main for you. It needs git-cliff in the project environment (`uv sync --group dev`). Before committing it pauses so you can add CHANGELOG notes (`--no-pause` commits straight away), and it never pushes. The manual commands below are the fallback. The script can't set variables in your shell, so afterwards run `version_number=$(uv version --short)` for steps 7-14.

### Steps
1. Start from upstream main with a clean working tree: `git switch main && git pull`. Step 6 commits every tracked change.
2. Pick the bump. Before 1.0 it is `patch`, or `minor` when a change since the last tag is breaking: a commit whose title carries `!` (`scripts/release.sh` lists them) or a PR in the draft release's Breaking changes section. Behaviour changes that are not marked breaking still get a note in step 5 and step 13. Read the new version: `version_number=$(uv version --bump <major|minor|patch> --dry-run --short)`
3. Create the release branch: `git switch --create=release-$version_number`
4. Bump the version: `uv version --bump <major|minor|patch> --no-sync`. This updates `pyproject.toml` and `uv.lock`.
5. Update the CHANGELOG: `uv run --no-sync git-cliff --github-repo xorq-labs/xorq -p CHANGELOG.md --tag v$version_number -u`. Add any extra notes (blog links, etc.) inside the new `## [$version_number]` section.
6. Commit: `git add --update && git commit -m "release: $version_number"`
7. Push the branch: `git push --set-upstream origin "release-$version_number"`. The push starts [ci-pre-release](https://github.com/xorq-labs/xorq/actions/workflows/ci-pre-release.yml).
8. Open a PR for `release-$version_number` titled `release: $version_number`. The title becomes the squash commit on main and keeps the PR out of the release notes.
9. ci-pre-release's `precheck` job checks that `uv.lock` is current, the version is not on PyPI, `CHANGELOG.md` has its section, and tag `v$version_number` does not exist. Once it passes, the `publish` job waits until dlovell, hussainsultan or mesejo approves the `pre-release` deployment on the run page. You can approve your own.
10. Wait for `publish` to upload the version to TestPyPI. The workflow runs no tests; the PR's CI does.
11. Check whether main moved since step 3: `git fetch origin && git log --oneline HEAD..origin/main`. If that lists commits, they would ship without a CHANGELOG entry. Run `git rebase origin/main && git checkout origin/main -- CHANGELOG.md`, redo step 5, then `git commit --amend --no-edit -a && git push --force-with-lease`. Reject the deployment that push asks for, since TestPyPI already has the version. When the PR's CI is green, "Squash and merge" it. `main` requires one approving review, which the maintainers on the bypass list of `main`'s branch protection skip with `gh pr merge --squash --admin`; every release PR so far has been merged that way. Anyone else needs an approval from someone other than the PR author. Do not pass `--delete-branch` to `gh pr merge`: the repository deletes the head branch on merge, and gh's local cleanup tries to remove a worktree that holds the branch.
12. Run `git fetch origin` and check that `git log -1 origin/main` is the release commit. If something merged after it, use the release commit's SHA in place of `origin/main`. Tag it and push that tag alone: `git tag v$version_number origin/main && git push origin v$version_number`. Never `git push --tags`, which publishes every local tag, including local-only archive or backup tags.
13. Publish the release. release-drafter keeps a draft release named `xorq unreleased` with the notes for everything merged since the last release. Review its notes first in the [releases page](https://github.com/xorq-labs/xorq/releases): the drafter lists every PR merged to `main`, so drop any merged after the release PR. Then set the tag and title to `v$version_number` and publish. Or from the command line:
    ```bash
    draft=$(gh api 'repos/xorq-labs/xorq/releases?per_page=100' --jq '[.[] | select(.draft and .name == "xorq unreleased") | .tag_name] | if length == 1 then .[0] else error("expected one draft named xorq unreleased, found \(length)") end') \
      && gh release edit "$draft" -R xorq-labs/xorq --tag "v$version_number" --title "v$version_number" --verify-tag --draft=false --latest
    ```
    The lookup stops unless exactly one such draft exists, and `--verify-tag` stops if step 12's tag is not on origin. Publishing starts `ci-release`, which uploads to PyPI.
14. Once ci-release passes, install the release from PyPI: `uv run --no-project --refresh-package xorq --with xorq==$version_number python -c 'import xorq; print(xorq.__version__)'` prints `$version_number`.

### When something goes wrong
A filename uploaded to PyPI or TestPyPI can never be uploaded again, even after the file is deleted. Everything else can be redone.

- If `precheck` fails, nothing was published and nobody was asked to approve. Fix it on the branch and push again.
- If the deployment was rejected or never approved, nothing was published. Re-run the workflow from the run page.
- If TestPyPI already has the version, precheck warns in the run log, and on the PR if one is open. `publish` skips files TestPyPI already has with the same content, so it can finish a partial upload, but TestPyPI rejects changed files. To publish changed content, close the PR, delete the branch (local and remote), and redo the manual steps from main with the next patch set explicitly: `version_number=<next>` in step 2 and `uv version $version_number --no-sync` in step 4. The script can't do this, because it bumps from main again.
- If you pushed the tag at the wrong commit, run `git push origin :refs/tags/v$version_number && git tag -d v$version_number` and repeat step 12. This is safe only while no published release uses the tag.
- If `ci-release` fails after the release is published, check PyPI. If the files got there, that version is spent and you cut a new patch release. If nothing was uploaded, re-run the failed job.
