#!/usr/bin/env bash
# The changed-lines style gate, shared by .github/workflows/ci-lint.yml and the
# xorq-check-style hook in .pre-commit-config.yaml so the two cannot drift.
# Arguments are the `git diff` revision selector:
# `origin/$GITHUB_BASE_REF...HEAD` in CI, `--cached` in the hook.
set -euo pipefail

# The same trees as LINT_PATHS in .github/workflows/ci-lint.yml and the
# ruff-check args in .pre-commit-config.yaml. None of the three can read the
# others, so test_lint_paths_agree in scripts/style_tests/ compares them and
# names the first that disagrees.
paths=(python examples docs scripts)

# The ratchet: the bulk of the style backlog, and with these enforced, touching
# any signature or parametrize list reddens the diff. See the whole-repo step in
# ci-lint.yml for why the ratchet is a flag here rather than pyproject config.
disable=type-annotations,pytest-param-id,future-annotations,print

# During conflict resolution the staged diff holds everything the merge brought
# in, not just the resolution. CI compares base...HEAD and never sees those
# lines, so checking them here only manufactures failures on other people's
# code.
if [ "${1-}" = "--cached" ] && [ -e "$(git rev-parse --git-dir)/MERGE_HEAD" ]; then
    exit 0
fi

# quotepath off: the default spells a non-ASCII path octal-escaped inside
# quotes, which the checker's `b/` prefix strip does not survive.
git -c core.quotepath=false diff "$@" -- "${paths[@]}" \
    | uv run --no-sync xorq-check-style --diff --disable "$disable"
