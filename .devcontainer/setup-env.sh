#!/usr/bin/env bash
# per-project: this entire script is project-specific — replace with your
# project's dependency installation and environment setup commands
set -euo pipefail

echo "Installing dependencies..."
uv sync --all-extras --all-groups
touch .venv/.last-sync

echo "Installing pre-commit hooks..."
uv run pre-commit install 2>/dev/null || true

if ! grep -q "direnv hook bash" ~/.bashrc 2>/dev/null; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
fi

mkdir -p ~/.config/direnv
cat > ~/.config/direnv/direnv.toml <<'TOML'
[whitelist]
prefix = ["/workspaces/src"]
TOML

if [ -f .envrcs/.envrc.user.template ] && [ ! -e .envrcs/.envrc.user ]; then
    cp .envrcs/.envrc.user.template .envrcs/.envrc.user
fi
