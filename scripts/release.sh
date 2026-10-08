#!/usr/bin/env bash
# Release Flow steps 1-6 (CONTRIBUTING.md): from a clean, up-to-date `main` to a
# release commit on `release-$version_number`. Run by hand; it never pushes.
#
# Usage: scripts/release.sh <major|minor|patch>
#
# git-cliff asks the GitHub API for PR links and contributor attribution; unauthenticated calls are rate-limited (60/hour):
#   export GITHUB_TOKEN=$(gh auth token)
set -euo pipefail

usage() {
    echo "usage: scripts/release.sh <major|minor|patch>" >&2
}

die() {
    echo "release.sh: error: $*" >&2
    exit 1
}

warn() {
    echo "release.sh: warning: $*" >&2
}

step() {
    echo "==> $*"
}

# ---------------------------------------------------------------- preflight --

bump_type="${1-}"
case "$bump_type" in
    major | minor | patch) ;;
    "")
        usage
        die "missing bump type"
        ;;
    *)
        usage
        die "unknown bump type '$bump_type' (expected major, minor or patch)"
        ;;
esac

command -v uv >/dev/null 2>&1 \
    || die "uv not found on PATH; see https://docs.astral.sh/uv/getting-started/installation/"

cd "$(git rev-parse --show-toplevel)"

# uv run falls back to PATH when git-cliff is missing from the project env, so require the resolved path to live under the env.
# One tab-separated line: a trailing empty field survives the newline stripping of $(...), a second line would not.
# shellcheck disable=SC2016  # the child shell expands these, not this one
cliff_probe=$(uv run --no-sync -- sh -c \
    'printf "%s\t%s\n" "${VIRTUAL_ENV-}" "$(command -v git-cliff || true)"') \
    || die "uv run --no-sync failed; is this a uv project checkout?"
IFS=$'\t' read -r project_env cliff_path <<<"$cliff_probe"
case "$cliff_path" in
    "$project_env"/*) ;;
    "")
        die "git-cliff is not installed in the project environment ($project_env); run 'uv sync --group dev' and retry"
        ;;
    *)
        die "git-cliff resolves to $cliff_path, outside the project environment ($project_env); the changelog must come from the pinned dev dependency, so run 'uv sync --group dev' and retry"
        ;;
esac

origin_url=$(git remote get-url origin 2>/dev/null) \
    || die "no remote named 'origin' (remotes: $(git remote | tr '\n' ' ')); the release flow pushes to 'origin'"
# A fork or mirror as 'origin' would make the sync below track its main, not upstream's, so match the whole URL.
case "${origin_url%.git}" in
    git@github.com:xorq-labs/xorq | ssh://git@github.com/xorq-labs/xorq | https://github.com/xorq-labs/xorq) ;;
    *)
        die "'origin' is $origin_url, not github.com/xorq-labs/xorq; a release must sync with and push to the upstream repository"
        ;;
esac

# Tracked changes only: they are all that `git add --update` can sweep into the release commit.
dirty=$(git status --porcelain --untracked-files=no)
[ -z "$dirty" ] \
    || die "working tree has tracked changes that 'git add --update' would sweep into the release commit:
$dirty
Commit or discard them first."

current_branch=$(git branch --show-current)
[ "$current_branch" = main ] \
    || die "on '${current_branch:-a detached HEAD}', not 'main'; run 'git switch main' first"

step "git fetch origin"
git fetch --quiet origin
read -r behind ahead < <(git rev-list --left-right --count origin/main...main)
if [ "$ahead" -gt 0 ] && [ "$behind" -gt 0 ]; then
    die "local main has diverged from origin/main ($ahead local-only, $behind remote-only commits); refusing to merge. Reset or rebase main first."
elif [ "$ahead" -gt 0 ]; then
    die "local main carries $ahead commit(s) not on origin/main; a release must start from the pushed main. Push or remove them first."
fi

# Before the dry run: the next version is read from pyproject.toml on main, which origin may have moved.
if [ "$behind" -gt 0 ]; then
    step "fast-forwarding main to origin/main ($behind commit(s))"
    git merge --quiet --ff-only origin/main
fi

if [ -z "${GITHUB_TOKEN-}" ]; then
    warn "GITHUB_TOKEN is not set: git-cliff will call the GitHub API unauthenticated (60 requests/hour) and may be rate-limited."
    warn "Ctrl-C now and 'export GITHUB_TOKEN=\$(gh auth token)', or continue without."
fi

next_version=$(uv --color never version --bump "$bump_type" --dry-run --short)
[ -n "$next_version" ] || die "uv version --bump $bump_type --dry-run printed no version"
branch="release-$next_version"

git show-ref --verify --quiet "refs/heads/$branch" \
    && die "branch '$branch' already exists; delete it (git branch -D $branch) or finish that release first"

# ---------------------------------------------------------------- mutations --

step "git switch --create $branch"
git switch --quiet --create "$branch"

# `uv version --bump` re-locks uv.lock itself; --no-sync only skips syncing the environment.
step "uv version --bump $bump_type"
uv --color never version --bump "$bump_type" --no-sync

version_number=$(uv --color never version --short)
export version_number
[ "$version_number" = "$next_version" ] \
    || die "version after bump is '$version_number' but the dry run said '$next_version'"
step "version_number=$version_number (exported; matches the dry run)"

step "git-cliff --tag v$version_number -u (prepending to CHANGELOG.md)"
uv run --no-sync git-cliff --github-repo xorq-labs/xorq -p CHANGELOG.md --tag "v$version_number" -u \
    || die "git-cliff failed. If the message mentions the GitHub API, 'export GITHUB_TOKEN=\$(gh auth token)' and rerun it by hand:
    uv run --no-sync git-cliff --github-repo xorq-labs/xorq -p CHANGELOG.md --tag v$version_number -u
then continue from step 6 of CONTRIBUTING.md on branch $branch."
grep -q "^## \[$version_number\]" CHANGELOG.md \
    || die "CHANGELOG.md has no '## [$version_number]' section after git-cliff; inspect it before committing"

echo
echo "CHANGELOG.md now has the [$version_number] section under the Unreleased placeholder."
printf 'Add any judgement-only notes (blog links, etc.) to CHANGELOG.md, then press Enter to commit: '
read -r _ || die "stdin closed before Enter; nothing committed. Branch $branch holds the bump and changelog."

step "git add --update && git commit -m \"release: $version_number\""
git add --update
git commit --quiet -m "release: $version_number"
git --no-pager log --oneline -1

cat <<MSG

Release commit made on $branch. Nothing has been pushed.
Inspect it (git show --stat), then push:

    git push --set-upstream origin $branch

Remaining manual steps (CONTRIBUTING.md, Release Flow):
  - open a PR for $branch
  - run ci-pre-release from $branch (Actions tab: Run workflow -> Branch $branch) and wait for it to pass
  - "Squash and merge" the PR
  - tag main and push that tag alone:
        git fetch origin && git tag v$version_number origin/main && git push origin v$version_number
  - publish the GitHub release at v$version_number to trigger the publishing workflow
MSG
