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
```
> [!IMPORTANT]
> Rename `.gitignore.template` to `.gitignore` 

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

### Steps
1. Ensure you're on upstream main: `git switch main && git pull`
2. Compute the new version number (`$version_number`) according to [Semantic Versioning](https://semver.org/) rules.
3. Create a branch that starts from the upstream main: `git switch --create=release-$version_number`
4. Update the version number in `pyproject.toml`: `version = "$version_number"`
5. Update the CHANGELOG using `git cliff --github-repo xorq-labs/xorq -p CHANGELOG.md --tag v$version_number -u`, manually add any additional notes (links to blogposts, etc.).
6. Create commit with a message denoting the release: `git add --update && git commit -m "release: $version_number"`.
7. Push the new branch: `git push --set-upstream upstream "release-$version_number"`
8. Open a PR for the new branch `release-$version_number`
9. Trigger the [ci-pre-release action](https://github.com/xorq-labs/xorq/actions/workflows/ci-pre-release.yml) from the branch created: Run workflow -> Use workflow from -> Branch `$version_number`
10. Wait for all ci-pre-release tests to pass
11. "Squash and merge" the PR
12. Tag the updated main with `v$version_number` and push the tag: `git fetch && git tag v$version_number origin/main && git push --tags`
13. Create a [GitHub release](https://github.com/xorq-labs/xorq/releases/new) to trigger the publishing workflow.
