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

# Symlink the committable post-checkout hook into .git/hooks/ so that every
# `git worktree add` — whether from dev/new-worktree, an agent, or a human —
# auto-locks the new worktree.  Refuses to clobber a non-symlink hook.
install_hooks() {
    local main
    main="$(dev_main_tree)" || return 1
    local hook="$main/dev/hooks/post-checkout"
    [ -f "$hook" ] || return 0

    # Unset core.hooksPath so git uses the default .git/hooks/.
    # Tools like Claude Code may set this to a linked-worktree path where
    # .git is a file (not a directory), breaking all hooks. pre-commit also
    # refuses to install when core.hooksPath is set.
    git config --unset-all core.hooksPath 2>/dev/null || true

    local hooks_dir="$main/.git/hooks"
    mkdir -p "$hooks_dir"
    if ! [ -e "$hooks_dir/post-checkout" ] || [ -L "$hooks_dir/post-checkout" ]; then
        ln -sf "../../dev/hooks/post-checkout" "$hooks_dir/post-checkout"
    fi
}
