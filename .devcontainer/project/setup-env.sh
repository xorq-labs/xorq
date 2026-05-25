#!/usr/bin/env bash
# Project-specific in-container setup. Runs inside the container as user
# vscode, invoked by dev/devcontainer.
#
# Subcommands:
#   first-run        — initial dependency install + dev tooling (after build)
#   sync-if-needed   — re-sync deps if the lockfile is newer than the stamp
set -euo pipefail

cmd="${1:-first-run}"

case "$cmd" in
    first-run)
        mkdir -p "$(dirname "${XORQ_LOG_PATH:-$HOME/.config/xorq/xorq.log}")"

        echo "Installing dependencies..."
        uv sync --all-extras --all-groups
        touch .venv/.last-sync

        # Worktrees inherit core.hooksPath from the main repo's .git/config;
        # pre-commit refuses to install when it's set. Override via env vars
        # (highest git-config precedence) to the worktree's own hooks dir.
        if git config core.hooksPath >/dev/null 2>&1; then
            _hooks_dir="$(git rev-parse --git-dir)/hooks"
            export GIT_CONFIG_COUNT=1
            export GIT_CONFIG_KEY_0=core.hooksPath
            export GIT_CONFIG_VALUE_0="${_hooks_dir}"
            if ! grep -q 'GIT_CONFIG_KEY_0=core.hooksPath' ~/.bashrc 2>/dev/null; then
                cat >> ~/.bashrc <<BASH
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=core.hooksPath
export GIT_CONFIG_VALUE_0="${_hooks_dir}"
BASH
            fi
        fi

        echo "Installing pre-commit hooks..."
        uv run pre-commit install 2>/dev/null || true

        if ! grep -q "direnv hook bash" ~/.bashrc 2>/dev/null; then
            echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
        fi

        mkdir -p ~/.config/direnv
        cat > ~/.config/direnv/direnv.toml <<TOML
[whitelist]
prefix = ["${PWD}"]
TOML

        if [ -f .envrcs/.envrc.user.template ] && [ ! -e .envrcs/.envrc.user ] && [ ! -L .envrcs/.envrc.user ]; then
            cp .envrcs/.envrc.user.template .envrcs/.envrc.user
        fi
        ;;
    sync-if-needed)
        [ -f uv.lock ] || exit 0
        lock_mtime="$(stat -c %Y uv.lock)"
        stamp_mtime="$(stat -c %Y .venv/.last-sync 2>/dev/null || echo 0)"
        if [ "$lock_mtime" -gt "$stamp_mtime" ]; then
            uv sync --all-extras --all-groups
            touch .venv/.last-sync
        fi
        ;;
    *)
        echo "usage: setup-env [first-run|sync-if-needed]" >&2
        exit 1
        ;;
esac
