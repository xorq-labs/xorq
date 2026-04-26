# Dev container

As an alternative to setting up a local environment, you can use the dev container. It works with both the main checkout and git worktrees.

**Prerequisites:** Docker, bash, git, and Python 3 on the host (the script must be run from inside a git repo). No other dependencies required (no Node, no uv). Linux x86_64 only — uses GNU coreutils and installs amd64 binaries.

```bash
# start the container (builds on first run)
devcontainer up

# open a shell inside the container
devcontainer exec

# run claude
devcontainer claude

# run claude with --dangerously-skip-permissions
devcontainer claude-dangerously-skip-permissions

# stop the container
devcontainer down

# destroy container and all volumes (venv, uv cache) to start fresh
devcontainer reset

# reset + remove images and host-side artifacts
devcontainer clean

# check whether the container is running
devcontainer status

# view container logs
devcontainer logs
```

## Worktrees

To use from a **worktree**, pass `-w` or run the script from the worktree directory:

```bash
# from the main checkout, targeting a worktree
devcontainer -w ../xorq-my-feature up

# or from inside the worktree (direnv adds dev/ to PATH)
devcontainer up
```

## Overriding project defaults

The scripts default to `PROJECT_NAME=xorq`. To override this (e.g., for a fork with a different name), copy the example and edit:

```bash
cp dev/project.env.example dev/project.env
# edit dev/project.env
```

This file is gitignored. Both `devcontainer` and `new-worktree` source it when present. The container workspace is always `/workspaces/src` and is not configurable.

**Note:** `exec` with no arguments opens a bash shell and requires an interactive terminal (TTY). With an explicit command (e.g., `exec pytest`), no TTY is needed.

## Tab completion

```bash
# bash
eval "$(devcontainer completions bash)"

# zsh
eval "$(devcontainer completions zsh)"

# fish
devcontainer completions fish | source
```

## `dev/devcontainer` vs `devcontainer.json`

This project has two ways to start the container:

1. **`dev/devcontainer`** (primary) — a shell script that calls `docker compose` directly. This is what the documentation above describes.
2. **`devcontainer.json`** — consumed by VS Code's "Reopen in Container" and the official `devcontainer` CLI.

The two paths diverge in what they provide:

| Capability | `dev/devcontainer` | `devcontainer.json` (VS Code) |
|---|---|---|
| Build & run | `docker compose` via the script | VS Code / `devcontainer` CLI |
| Toolchain (uv, just, sops, gh, node, claude) | Dockerfile — always applied | Dockerfile — always applied |
| UID/GID matching | `DEV_UID`/`DEV_GID` build args | Not applied (uses image defaults) |
| Git config, gh auth, Claude setup | `setup_git`, `setup_gh`, `setup_claude` in the script | Not applied |
| Worktree support | Full (mount host `.git`, resolve worktree paths) | Not supported |
| SSH agent forwarding | Mounted via compose env vars | Not applied |
| sops age keys | Mounted read-only via compose | Not applied |
| Port forwarding | Not handled (use `docker compose` ports) | `forwardPorts` in `devcontainer.json` |
| VS Code extensions & settings | Not applied | `customizations.vscode` in `devcontainer.json` |
| Dep sync on lockfile change | `sync_if_needed` in the script | Not applied |
| Image staleness check | Checks Dockerfile/compose mtime | Handled by VS Code |

`devcontainer.json` is kept for contributors who prefer the VS Code flow, but `dev/devcontainer` is the supported path — it handles worktrees, credential forwarding, and Claude isolation that `devcontainer.json` does not.

## Claude isolation & permission audit

The container's `~/.claude` is isolated from the host. On each `up`, the host's credentials and permission baseline are copied in read-only, so the container inherits your current permissions but can never modify host settings, hooks, or session history. Host hooks (global and project-level) are intentionally not copied — they reference host paths and binaries that don't exist inside the container. Only permissions are inherited.

A `PreToolUse` hook logs every tool invocation to `.claude/container-audit/audit.jsonl` in the workspace. Use the `audit` subcommand to review:

```bash
# summary: tool counts, bash prefixes, permissions not in host baseline
devcontainer audit

# just the new permission patterns (for piping into settings)
devcontainer audit --new

# all observed patterns (including those already in baseline)
devcontainer audit --all

# clear the audit log
devcontainer audit --clear
```

Container session logs are written to `.claude/container-sessions/` in the workspace, visible from the host.

> [!NOTE]
> **Auto-rebuild scope:** The script detects changes to `Dockerfile` and `docker-compose.yml` and prompts to rebuild. Changes to other files in `.devcontainer/` (e.g. `setup-claude.py`, `audit-hook`, `setup-env.sh`) are baked into the image at build time and require `devcontainer reset` followed by `devcontainer up` to take effect.
>
> **Volume persistence:** `down` stops the container but leaves Docker volumes intact, including the `claude-home` volume which contains copied credentials. Use `reset` to destroy all volumes.
>
> **Host git access:** The host's `.git` directory is mounted read-write inside the container (required for git operations in worktrees). A container compromise could modify host git history, hooks, and refs.
