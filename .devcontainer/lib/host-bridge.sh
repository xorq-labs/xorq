# Host-to-container bridge functions: SSH agent forwarding, host git config,
# gh credentials, and Claude config setup.
#
# Sourced by dev/devcontainer. The sourcer is expected to provide:
#   - dc(...)         — `docker compose ... -p $CONTAINER_NAME` wrapper
#   - dc_exec(...)    — `dc exec` with the standard env vars
#   - is_running()    — boolean check that the app service is up
# and to export:
#   - CONTAINER_NAME           — used to derive a per-container SSH forward port
#   - CONTAINER_WORKSPACE      — passed through to setup-claude
#   - DEV_WORKSPACE            — host path of the workspace, used as project key
#   - HAS_SOCAT                — set to "true" if socat is on PATH

SSH_FORWARD_PIDFILE="/tmp/devcontainer-ssh-forward-${CONTAINER_NAME}.pid"

ssh_forward_port() {
    local hash
    hash="$(echo "$CONTAINER_NAME" | cksum | cut -d' ' -f1)"
    echo $(( (hash % 16384) + 49152 ))
}

stop_ssh_forward() {
    if [ -f "$SSH_FORWARD_PIDFILE" ]; then
        local pid
        pid="$(cat "$SSH_FORWARD_PIDFILE")"
        kill "$pid" 2>/dev/null || true
        rm -f "$SSH_FORWARD_PIDFILE"
    fi
    if is_running; then
        dc_exec bash -c 'pkill -f "socat UNIX-LISTEN:/run/ssh-agent/" 2>/dev/null' || true
    fi
}

setup_ssh_forward() {
    if [ "${HAS_SOCAT:-false}" != "true" ]; then
        echo "warning: socat not found on host — SSH agent forwarding disabled (apt install socat)" >&2
        return 0
    fi
    if [ ! -S "${SSH_AUTH_SOCK:-}" ]; then
        echo "warning: no SSH agent detected — git over SSH won't work inside the container" >&2
        return 0
    fi

    stop_ssh_forward

    local port
    port="$(ssh_forward_port)"

    local bridge_ip
    bridge_ip="$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')"
    if [ -z "$bridge_ip" ]; then
        echo "error: could not determine Docker bridge gateway IP" >&2
        return 1
    fi

    socat "TCP-LISTEN:${port},bind=${bridge_ip},reuseaddr,fork" \
          "UNIX-CONNECT:${SSH_AUTH_SOCK}" &
    local host_pid=$!
    echo "$host_pid" > "$SSH_FORWARD_PIDFILE"

    sleep 0.2
    if ! kill -0 "$host_pid" 2>/dev/null; then
        echo "error: host-side SSH forwarder failed to start (port $port may be in use)" >&2
        rm -f "$SSH_FORWARD_PIDFILE"
        return 1
    fi

    dc_exec bash -c "
        nohup socat \
            UNIX-LISTEN:/run/ssh-agent/agent.sock,fork,unlink-early,mode=600 \
            TCP:host.docker.internal:${port} \
            </dev/null >/dev/null 2>&1 &
    "

    sleep 0.3
    if ! dc_exec ssh-add -l >/dev/null 2>&1; then
        echo "warning: SSH agent bridge started but ssh-add -l failed inside the container" >&2
    fi
}

setup_git() {
    local name email
    name="$(git config user.name 2>/dev/null || true)"
    email="$(git config user.email 2>/dev/null || true)"
    if [ -n "$name" ]; then
        dc_exec git config --global user.name "$name"
    fi
    if [ -n "$email" ]; then
        dc_exec git config --global user.email "$email"
    fi
}

setup_gh() {
    local hosts="$HOME/.config/gh/hosts.yml"
    if [ -f "$hosts" ]; then
        dc_exec mkdir -p /home/vscode/.config/gh
        docker cp "$hosts" "$(dc ps -q app)":/home/vscode/.config/gh/hosts.yml
    fi
}

setup_claude_credentials() {
    local cred_dir="$HOME/.claude/credentials"
    local cred_file="$HOME/.claude/.credentials.json"

    mkdir -p "$cred_dir"

    # First-time migration: move legacy credential file into the shared
    # directory and leave a symlink.  The guard also repairs if anything
    # (e.g. a future atomic-rename writer) ever replaces the symlink.
    if [ -f "$cred_file" ] && [ ! -L "$cred_file" ]; then
        mv "$cred_file" "$cred_dir/.credentials.json"
        ln -s credentials/.credentials.json "$cred_file"
    fi
}

setup_claude() {
    local host_project_key container_project_key
    host_project_key="$(echo "$DEV_WORKSPACE" | sed 's|/|-|g')"
    container_project_key="$(echo "$CONTAINER_WORKSPACE" | sed 's|/|-|g')"

    # named volume root is owned by root; fix so vscode can write
    dc exec -u root app chown -R vscode:vscode /home/vscode/.claude

    dc exec \
        -e DEV_CONTAINER_WORKSPACE="$CONTAINER_WORKSPACE" \
        -e DEV_HOST_PROJECT_KEY="$host_project_key" \
        -e DEV_CONTAINER_PROJECT_KEY="$container_project_key" \
        app python3 /usr/local/bin/setup-claude
}
