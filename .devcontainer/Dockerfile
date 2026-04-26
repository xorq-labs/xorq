# per-project: base image
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ARG USER_UID=1000
ARG USER_GID=1000
ARG HOST_USER=
ARG CONTAINER_WORKSPACE

RUN if [ "$USER_GID" != "1000" ]; then groupmod -g $USER_GID vscode; fi \
    && if [ "$USER_UID" != "1000" ]; then usermod -u $USER_UID vscode; fi \
    && chown -R $USER_UID:$USER_GID /home/vscode

# per-project: system packages and toolchain
# Node 20 LTS — EOL 2026-10-30, upgrade to setup_22.x when available
ARG NODE_MAJOR=20
ARG NODESOURCE_SHA256=2c4c6683a17b6f4128898a7b521e3c8bb725a99ffaf1b5e32ac97c6fa7d381be
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x -o /tmp/nodesource.sh \
    && echo "$NODESOURCE_SHA256  /tmp/nodesource.sh" | sha256sum -c - \
    && bash /tmp/nodesource.sh \
    && rm /tmp/nodesource.sh \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        direnv \
        nodejs \
        gh \
    && rm -rf /var/lib/apt/lists/*

# per-project: uv (Python package manager)
ARG UV_VERSION=0.7.8
ARG UV_INSTALLER_SHA256=3e3043ca08e1156fbe18d90a1a4def3ae795418857c8f4ed3f807ffc45e51c3d
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh -o /tmp/uv-install.sh \
    && echo "$UV_INSTALLER_SHA256  /tmp/uv-install.sh" | sha256sum -c - \
    && env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh \
    && rm /tmp/uv-install.sh

# per-project: just (task runner)
ARG JUST_VERSION=1.40.0
ARG JUST_INSTALLER_SHA256=e68e568671780281964fe61bbe0396507fd690a7677fc5a42fcdc69e737dd54a
RUN curl -sSf https://just.systems/install.sh -o /tmp/just-install.sh \
    && echo "$JUST_INSTALLER_SHA256  /tmp/just-install.sh" | sha256sum -c - \
    && bash /tmp/just-install.sh --to /usr/local/bin --tag $JUST_VERSION \
    && rm /tmp/just-install.sh

# per-project: sops (secrets management)
ARG SOPS_VERSION=3.9.4
ARG SOPS_SHA256=5488e32bc471de7982ad895dd054bbab3ab91c417a118426134551e9626e4e85
RUN curl -LsSf https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64 \
        -o /usr/local/bin/sops \
    && echo "$SOPS_SHA256  /usr/local/bin/sops" | sha256sum -c - \
    && chmod +x /usr/local/bin/sops

ARG CLAUDE_CODE_VERSION=2.1.119
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

COPY setup-claude.py /usr/local/bin/setup-claude
COPY audit-hook /usr/local/bin/audit-hook
COPY setup-env.sh /usr/local/bin/setup-env
RUN chmod +x /usr/local/bin/setup-claude /usr/local/bin/audit-hook /usr/local/bin/setup-env

# per-project: cache directories for build tools
RUN mkdir -p /home/vscode/.cache/uv /home/vscode/.ssh \
    && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.ssh \
    && chmod 700 /home/vscode/.ssh

RUN HOST_USER="$(basename "$HOST_USER")" && \
    if [ -n "$HOST_USER" ] && [ "$HOST_USER" != "vscode" ]; then \
        ln -s /home/vscode "/home/$HOST_USER"; \
    fi

# per-project: PATH additions for project tooling
ENV PATH="${CONTAINER_WORKSPACE}/.venv/bin:$PATH"
ENV HOME=/home/vscode
