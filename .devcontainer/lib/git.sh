# Resolve the main worktree path. The first entry in `git worktree list
# --porcelain` is always the main checkout, regardless of where the command
# is run from. Worktrees that need main-only state (project.env, dev/ scripts)
# must use this resolver, not `git rev-parse --show-toplevel` (which returns
# the *current* worktree).

dev_main_tree() {
    local out
    if ! out=$(git worktree list --porcelain 2>/dev/null); then
        echo "error: not in a git repository (run from within a checkout)" >&2
        return 1
    fi
    printf '%s\n' "$out" | head -1 | sed 's/^worktree //'
}
