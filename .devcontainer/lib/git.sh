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
    local dest="$main/.git/hooks/post-checkout"
    [ -f "$hook" ] || return 0
    if [ -e "$dest" ] && ! [ -L "$dest" ]; then
        return 0
    fi
    ln -sf "../../dev/hooks/post-checkout" "$dest"
}
