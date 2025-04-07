## Contributing Guide

### Setting up a development environment

#### Using pip and venv

```bash
# fetch this repo
git clone git@github.com:xorq-labs/xorq.git
# prepare development environment (used to build wheel / install in development)
python3 -m venv venv
# activate the venv
source venv/bin/activate
# update pip itself if necessary
python -m pip install -U pip
# install dependencies 
python -m pip install -r requirements-dev.txt
# install current package in editable mode
python -m pip install -e .
# set up the git hook scripts
pre-commit install
```

#### Using uv

This assumes you have uv installed, otherwise please follow these [instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# fetch this repo
git clone git@github.com:xorq-labs/xorq.git
# set uv run command to not sync 
export UV_NO_SYNC=1
# prepare development environment and install dependencies (including current package)
uv sync --all-extras --all-groups
# activate the venv
source venv/bin/activate
# set up the git hook scripts
uv run pre-commit install
```

### Running the test suite
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

To test the code:
```bash
# make sure you activate the venv using "source venv/bin/activate" first
just up postgres # some of the tests use postgres
python -m pytest # or pytest
```

### Working with xorq-datafusion

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

Notice that that our test run with `--no-sources` so they will fail if a new version of `xorq-datafusion` with the required 
change is not present in PyPI. 

### Writing the commit

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


### Release Flow
***This section is intended for xorq maintainers***

#### Steps
1. Ensure you're on upstream main: `git switch main && git pull`
2. Compute the new version number (`$version_number`) according to [Semantic Versioning](https://semver.org/) rules.
3. Create a branch that starts from the upstream main: `git switch --create=release-$version_number`
4. Update the version number in `pyproject.toml`: `version = "$version_number"`
5. Update the CHANGELOG using `git cliff --github-repo xorq-labs/xorq -p CHANGELOG.md --tag v$version_number -u`, manually add any additional notes (links to blogposts, etc.).
6. Create commit with message denoting the release: `git add --update && git commit -m "release: $version_number"`.
7. Push the new branch: `git push --set-upstream upstream "release-$version_number"`
8. Open a PR for the new branch `release-$version_number`
9. Trigger the [ci-pre-release action](https://github.com/xorq-labs/xorq/actions/workflows/ci-pre-release.yml) from the branch created: Run workflow -> Use workflow from -> Branch `$version_number`
10. Wait for the ci-pre-release tests to all pass
11. "Squash and merge" the PR
12. Tag the updated main with `v$version_number` and push the tag: `git fetch && git tag v$version_number origin/main && git push --tags`
13. Create a [GitHub release](https://github.com/xorq-labs/xorq/releases/new) to trigger the publishing workflow.
