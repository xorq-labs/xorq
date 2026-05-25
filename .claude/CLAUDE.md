## Devcontainer safety

This repo is often worked on from inside a devcontainer. The `.git` directory is shared between the host and the container.

Worktrees created via `dev/new-worktree` are automatically locked to prevent `git worktree add` (which runs an implicit
prune) from destroying their metadata when paths are unreachable from inside a container. `dev/cleanup-worktree` unlocks
before removing. Always use these scripts to create and remove worktrees.
