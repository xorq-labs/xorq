# Dev container

As an alternative to setting up a local environment, you can use the dev container. It works with both the main checkout and git worktrees.

**Prerequisites (host):** Docker, bash, git, and Python 3. No other dependencies required (no Node, no uv). The script must be run from inside a git repo.

> [!IMPORTANT]
> **Linux x86_64 only.** Installs amd64 binaries and uses GNU coreutils. macOS and Windows are not supported.

**Toolchain (container):** Python 3.12 with uv 0.7.8, just 1.40.0, sops 3.9.4, gh, direnv, Node 20, and Claude Code 2.1.119. Versions are pinned in `.devcontainer/Dockerfile`.

**Setup:** The `devcontainer` script lives in `dev/`. Run `direnv allow` in the repo root to add it to your PATH, or invoke it directly as `./dev/devcontainer`.

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

**Notes:**
- After the first `up`, subsequent `exec` and `claude` invocations auto-start the container if it isn't already running.
- The container rebuilds automatically when the `Dockerfile` or compose config changes.
- `down` prompts before stopping a running container; `reset` prompts before destroying volumes.
- `exec` with no arguments opens a bash shell and requires an interactive TTY. With an explicit command (e.g. `exec uv run pytest -m core`), no TTY is needed.

## Worktrees

To use from a **worktree**, pass `-w` or run the script from the worktree directory:

```bash
# from the main checkout, targeting a worktree
devcontainer -w ../xorq-my-feature up

# or from inside the worktree (direnv adds dev/ to PATH)
devcontainer up
```

## Project configuration

All editable bits live in **`.devcontainer/project/`**. Everything outside that directory (the Dockerfile, `docker-compose.yml`, `dev/devcontainer`, `audit-report.py`, etc.) is generic and can be copied unchanged into another project.

| File | Role |
|---|---|
| `install-system.sh` | apt packages and language toolchain (runs as root during `docker build`) |
| `setup-env.sh` | first-run + sync-on-lockfile-change hooks (runs in-container as `vscode`) |
| `compose.override.yml` | extra named volumes, env vars, and the `EXTRA_PATH` build arg |
| `worktree-symlinks.txt` | paths under the main worktree to symlink into new worktrees |
| `worktree-copies.txt` | paths to copy (not symlink) into new worktrees; globs allowed |
| `audit-prefixes.txt` | bash command prefixes that should be grouped by their first two words |
| `project.env.example` | template for the gitignored `project.env` overrides |

Copy `project.env.example` to `project.env` (gitignored) to set:

- `PROJECT_NAME` — overrides the per-project namespace (defaults to the basename of the main checkout). Scopes the container name and shared docker volumes (`<PROJECT_NAME>-uv-cache`, etc.).
- `MODEL_VERSION` — pins the Claude model inside the container (written into `~/.claude/settings.json` as `model`). Leave unset to use Claude Code's default.

The container workspace is always `/workspaces/src` — threaded through compose, the Dockerfile, and setup-claude as `DEV_CONTAINER_WORKSPACE`, but changing it is not supported.

## Adapting to another project

Drop `.devcontainer/` and `dev/` into your project, then edit `.devcontainer/project/`:

1. **`install-system.sh`** — apt packages and language toolchain. The default installs `build-essential libpq-dev direnv` and `uv` for a Python project. Replace with whatever your project needs (e.g. `golang-go`, `rustup`, `bun`).
2. **`setup-env.sh`** — what runs after the container starts. The default does `uv sync`, installs pre-commit hooks, sets up `direnv`, and seeds `.envrcs/.envrc.user`. Both subcommands (`first-run`, `sync-if-needed`) are called by `dev/devcontainer`; keep their interface and replace the bodies.
3. **`compose.override.yml`** — named volumes, host-service env vars, and the `EXTRA_PATH` build arg (defaults to the Python venv's bin dir). Delete or rename volumes you don't need.
4. **`worktree-symlinks.txt`** / **`worktree-copies.txt`** — what `setup-worktree` propagates from the main worktree.
5. **`audit-prefixes.txt`** — bash commands that should be grouped two-words-deep in the audit report.
6. **`devcontainer.json`** (one level up) — VS Code's entry point. Edit `name`, `forwardPorts`, and `customizations.vscode` for your project. This is the only editable file outside `project/`; it can't import sub-files because of the devcontainer.json spec.
7. **`Dockerfile`** — only edit `BASE_IMAGE` (default `mcr.microsoft.com/devcontainers/python:3.12-bookworm`) if you need a non-Python base image. Override via a build arg in `compose.override.yml` rather than editing the Dockerfile in place.

If `install-system.sh` is empty or `compose.override.yml` is missing, the container still builds — they're optional layers on top of the generic image.

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
| Image staleness check | Content hash of Dockerfile, compose, and COPY'd files | Handled by VS Code |

## Troubleshooting

- **`devcontainer: command not found`** — `dev/` isn't on PATH. Run `direnv allow` in the repo root, or invoke as `./dev/devcontainer`.
- **`docker: command not found` or "Cannot connect to the Docker daemon"** — Docker isn't installed or the daemon isn't running. Start Docker (`systemctl --user start docker` or your distro's equivalent).
- **`docker compose` reports "unknown command"** — you have Compose v1. Install Compose v2 (the script uses `docker compose`, not `docker-compose`).
- **Build fails partway through `up`** — inspect with `devcontainer logs`, then `devcontainer clean && devcontainer up` to rebuild from scratch.
- **Files in `.venv/` owned by root, or `Permission denied` writing to the workspace** — UID/GID drift between the host and the image. `devcontainer clean && devcontainer up` rebuilds with the current host UID/GID.
- **`uv sync` runs every entry** — the lockfile is newer than `.venv/.last-sync`. Expected after `git pull`; harmless.
- **Auto-rebuild prompt won't go away** — the content hash of `Dockerfile` / compose changed. Accept the rebuild, or `devcontainer clean` to reset state.

## Claude isolation & permission audit

The container's `~/.claude` is isolated from the host. On each entry (`up`, `exec`, `claude`), `setup-claude` re-copies the following from the host so the container reflects your current host configuration:

- **Credentials** — `~/.claude/.credentials.json`
- **Global permissions and `CLAUDE.md`** — the `permissions` block from `~/.claude/settings.json`, plus `~/.claude/CLAUDE.md`
- **Project permissions and memory** — the `permissions` block from `~/.claude/projects/<host-project-key>/settings.json` and `settings.local.json`, plus the project's `memory/` directory

Container-side Claude settings changes (e.g. permissions granted mid-session) are overwritten on the next entry. Host hooks are intentionally **not** copied — they reference host paths and binaries that don't exist inside the container. Only permissions, memory, and credentials cross the boundary.

A `PreToolUse` hook logs every tool invocation to `.claude/container-audit/audit.jsonl` in the workspace. Use the `audit` subcommand to review:

```bash
# summary (default): tool counts, bash prefixes, permissions not in host baseline
devcontainer audit
devcontainer audit --summary

# just the new permission patterns (for piping into settings)
devcontainer audit --new

# all observed patterns (including those already in baseline)
devcontainer audit --all

# clear the audit log
devcontainer audit --clear
```

Container session logs are written to `.claude/container-sessions/` in the workspace, visible from the host.

> [!NOTE]
> **Volume persistence:** `down` stops the container but leaves Docker volumes intact, including the `claude-home` volume which contains copied credentials (API keys, gh token). Use `reset` to destroy all volumes and scrub credentials.
>
> **Host git access:** The host's `.git` directory is mounted read-write inside the container (required for git operations in worktrees). A container compromise could modify host git history, hooks, and refs.
