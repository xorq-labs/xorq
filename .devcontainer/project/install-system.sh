#!/usr/bin/env bash
# Project-specific system packages and language toolchain.
# Runs as root during `docker build`, after the generic Dockerfile installs
# the infrastructure layer (Node + claude-code, gh, socat, just, sops).
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    direnv
rm -rf /var/lib/apt/lists/*

# uv (Python package manager)
UV_VERSION=0.7.8
UV_INSTALLER_SHA256=3e3043ca08e1156fbe18d90a1a4def3ae795418857c8f4ed3f807ffc45e51c3d
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv-install.sh
echo "$UV_INSTALLER_SHA256  /tmp/uv-install.sh" | sha256sum -c -
env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh
rm /tmp/uv-install.sh
