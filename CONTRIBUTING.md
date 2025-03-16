## Contributing Guide

### Setting up a development environment

This assumes that you have rust and cargo installed. We use the workflow recommended by [pyo3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

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
# set up the git hook scripts
pre-commit install
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
maturin develop
just up postgres # some of the tests use postgres
python -m pytest # or pytest
```

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
4. Update the version number in `Cargo.toml`: `version = "$version_number"`
5. Update the CHANGELOG using `git cliff --github-repo xorq-labs/xorq -p CHANGELOG.md --tag v$version_number -u`, manually add any additional notes (links to blogposts, etc.).
6. Create commit with message denoting the release: `git add --update && git commit -m "release: $version_number"`.
7. Push the new branch: `git push --set-upstream origin "release-$version_number"`
7. Open a PR for the new branch `release-$version_number`
8. Trigger the [ci-pre-release action](https://github.com/xorq-labs/xorq/actions/workflows/ci-pre-release.yml) from the branch created: Run workflow -> Use workflow from -> Branch `$version_number`
9. Wait for the ci-pre-release tests to all pass
10. "Squash and merge" the PR
11. Tag the updated main with `v$version_number` and push the tag: `git fetch && git tag v$version_number origin/main && git push --tags`
12. Create a [GitHub release](https://github.com/xorq-labs/xorq/releases/new) to trigger the publishing workflow.
