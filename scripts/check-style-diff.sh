#!/usr/bin/env bash
# The changed-lines style gate, shared by .github/workflows/ci-lint.yml and the
# xorq-check-style hook in .pre-commit-config.yaml so the two cannot drift.
# Arguments are the `git diff` revision selector: `origin/main...HEAD` in CI,
# `--cached` in the hook.
set -euo pipefail

# Keep in step with LINT_PATHS in .github/workflows/ci-lint.yml, which scopes
# Ruff and the whole-repo gate to the same trees.
paths=(python examples docs scripts)

# The ratchet: these four are the bulk of the style backlog, and with them on,
# touching any signature or parametrize list reddens the diff. They are passed
# here rather than set in [tool.xorq-style] disable because that would apply to
# every invocation, including the editor PostToolUse hook -- the rules would
# then go unreported on new code as well as old, and their counts could never
# fall.
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
