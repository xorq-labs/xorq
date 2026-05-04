# Generic devcontainer Dockerfile.
# Project-specific edits live in project/install-system.sh, which is COPY'd
# in and executed near the bottom. To target a different language/runtime,
# override BASE_IMAGE via docker-compose build args.
ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/python:3.12-bookworm
FROM ${BASE_IMAGE}

ARG USER_UID=1000
ARG USER_GID=1000
ARG HOST_USER=
ARG DEV_CONTAINER_WORKSPACE
ARG EXTRA_PATH=

RUN if [ "$USER_GID" != "1000" ]; then groupmod -g $USER_GID vscode; fi \
    && if [ "$USER_UID" != "1000" ]; then usermod -u $USER_UID vscode; fi \
    && chown -R $USER_UID:$USER_GID /home/vscode

# Generic infrastructure: Node (for claude-code), gh, socat (SSH agent bridge)
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
        nodejs \
        gh \
        socat \
    && rm -rf /var/lib/apt/lists/*

# just (task runner)
ARG JUST_VERSION=1.40.0
ARG JUST_INSTALLER_SHA256=e68e568671780281964fe61bbe0396507fd690a7677fc5a42fcdc69e737dd54a
RUN curl -sSf https://just.systems/install.sh -o /tmp/just-install.sh \
    && echo "$JUST_INSTALLER_SHA256  /tmp/just-install.sh" | sha256sum -c - \
    && bash /tmp/just-install.sh --to /usr/local/bin --tag $JUST_VERSION \
    && rm /tmp/just-install.sh

# sops (secrets management)
ARG SOPS_VERSION=3.9.4
ARG SOPS_SHA256=5488e32bc471de7982ad895dd054bbab3ab91c417a118426134551e9626e4e85
RUN curl -LsSf https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64 \
        -o /usr/local/bin/sops \
    && echo "$SOPS_SHA256  /usr/local/bin/sops" | sha256sum -c - \
    && chmod +x /usr/local/bin/sops

ARG CLAUDE_CODE_VERSION=2.1.119
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Project-specific system packages and language toolchain.
COPY project/install-system.sh /tmp/install-system.sh
RUN bash /tmp/install-system.sh && rm /tmp/install-system.sh

COPY setup-claude.py /usr/local/bin/setup-claude
COPY audit-hook /usr/local/bin/audit-hook
COPY project/setup-env.sh /usr/local/bin/setup-env
RUN chmod +x /usr/local/bin/setup-claude /usr/local/bin/audit-hook /usr/local/bin/setup-env

# /home/vscode/.claude/.credentials.json is created here as a symlink into
# the credentials/ bind-mount (declared in compose.yml). Image-baked so it
# lands in the per-worktree claude-home named volume on first init; the
# host-side migration in lib/host-bridge.sh handles the legacy file move.
RUN mkdir -p /home/vscode/.cache /home/vscode/.ssh /home/vscode/.claude \
    && ln -s credentials/.credentials.json /home/vscode/.claude/.credentials.json \
    && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.ssh /home/vscode/.claude \
    && chown -h vscode:vscode /home/vscode/.claude/.credentials.json \
    && chmod 700 /home/vscode/.ssh

RUN HOST_USER="$(basename "$HOST_USER")" && \
    if [ -n "$HOST_USER" ] && [ "$HOST_USER" != "vscode" ]; then \
        ln -s /home/vscode "/home/$HOST_USER"; \
    fi

ENV PATH="${EXTRA_PATH}${EXTRA_PATH:+:}${PATH}"
ENV HOME=/home/vscode
