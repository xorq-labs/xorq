# Host-to-container bridge functions: SSH agent forwarding, GPG agent
# forwarding, host git config, gh credentials, and Claude config setup.
#
# Sourced by dev/devcontainer. The sourcer is expected to provide:
#   - dc(...)         — `docker compose ... -p $DEV_CONTAINER_NAME` wrapper
#   - dc_exec(...)    — `dc exec` with the standard env vars
#   - is_running()    — boolean check that the app service is up
# and to set (script-locals are fine — this lib is sourced, not exec'd):
#   - DEV_CONTAINER_NAME       — used to derive per-container forward ports
#   - DEV_CONTAINER_WORKSPACE  — passed through to setup-claude
#   - DEV_WORKSPACE            — host path of the workspace, used as project key
#   - DEV_HAS_SOCAT            — set to "true" if socat is on PATH

DEV_SSH_FORWARD_PIDFILE="/tmp/devcontainer-ssh-forward-${DEV_CONTAINER_NAME}.pid"

ssh_forward_port() {
    local hash
    hash="$(echo "$DEV_CONTAINER_NAME" | cksum | cut -d' ' -f1)"
    echo $(( (hash % 16384) + 49152 ))
}

stop_ssh_forward() {
    if [ -f "$DEV_SSH_FORWARD_PIDFILE" ]; then
        local pid
        pid="$(cat "$DEV_SSH_FORWARD_PIDFILE")"
        kill "$pid" 2>/dev/null || true
        rm -f "$DEV_SSH_FORWARD_PIDFILE"
    fi
    # Catch orphans whose PID file was lost (tmpfiles cleanup, manual rm, etc.)
    # Pattern includes the full socat arg structure to avoid killing unrelated
    # socat processes that happen to use the same port.
    local port
    port="$(ssh_forward_port)"
    pkill -f "socat TCP-LISTEN:${port},bind=.+,reuseaddr,fork UNIX-CONNECT:" 2>/dev/null || true
    if is_running; then
        dc_exec bash -c 'pkill -f "socat UNIX-LISTEN:/run/ssh-agent/" 2>/dev/null' || true
    fi
}

setup_ssh_forward() {
    if [ "${DEV_HAS_SOCAT:-false}" != "true" ]; then
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

    # host.docker.internal inside the container reaches this host, but the
    # address socat must bind to depends on the Docker runtime:
    #   Linux-native:  host-gateway == bridge gateway (a real host IP)
    #   Docker Desktop: host.docker.internal routes to host loopback
    #   WSL2 + Desktop: same as Docker Desktop (bridge IP is VM-internal)
    local bind_addr
    if [ "$(uname -s)" = "Linux" ] && ! grep -qiF microsoft /proc/version 2>/dev/null; then
        bind_addr="$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)" || true
        if [ -z "$bind_addr" ]; then
            echo "error: could not determine Docker bridge gateway IP — is the bridge network enabled?" >&2
            return 1
        fi
    else
        bind_addr="127.0.0.1"
    fi

    socat "TCP-LISTEN:${port},bind=${bind_addr},reuseaddr,fork" \
          "UNIX-CONNECT:${SSH_AUTH_SOCK}" &
    local host_pid=$!
    echo "$host_pid" > "$DEV_SSH_FORWARD_PIDFILE"

    # Wait briefly for socat to fail-fast on bad args (e.g. port in use); if
    # it's still alive after the window, treat it as healthy. Bounded retry
    # instead of a fixed sleep — flaky on slow hosts.
    local i
    for i in $(seq 1 50); do
        sleep 0.05
        kill -0 "$host_pid" 2>/dev/null || break
    done
    if ! kill -0 "$host_pid" 2>/dev/null; then
        echo "error: host-side SSH forwarder failed to start (port $port may be in use)" >&2
        rm -f "$DEV_SSH_FORWARD_PIDFILE"
        return 1
    fi

    # -d (detached) is required: backgrounding via `bash -c "... &"` doesn't
    # survive the exec session teardown — Docker SIGTERMs the process tree
    # when the exec call returns, killing socat despite nohup.
    dc exec -d app socat \
        UNIX-LISTEN:/run/ssh-agent/agent.sock,fork,unlink-early,mode=600 \
        TCP:host.docker.internal:${port}

    # Poll ssh-add -l until the bridge is reachable through the container.
    for i in $(seq 1 30); do
        if dc_exec ssh-add -l >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    echo "error: SSH agent bridge started but ssh-add -l never succeeded inside the container" >&2
    return 1
}

## GPG agent forwarding ######################################################

DEV_GPG_FORWARD_PIDFILE="/tmp/devcontainer-gpg-forward-${DEV_CONTAINER_NAME}.pid"

gpg_forward_port() {
    local hash
    hash="$(echo "${DEV_CONTAINER_NAME}-gpg" | cksum | cut -d' ' -f1)"
    echo $(( (hash % 16384) + 49152 ))
}

stop_gpg_forward() {
    if [ -f "$DEV_GPG_FORWARD_PIDFILE" ]; then
        local pid
        pid="$(cat "$DEV_GPG_FORWARD_PIDFILE")"
        kill "$pid" 2>/dev/null || true
        rm -f "$DEV_GPG_FORWARD_PIDFILE"
    fi
    local port
    port="$(gpg_forward_port)"
    pkill -f "socat TCP-LISTEN:${port},bind=.+,reuseaddr,fork UNIX-CONNECT:" 2>/dev/null || true
    if is_running; then
        dc_exec bash -c 'pkill -f "socat UNIX-LISTEN:/run/gpg-agent/" 2>/dev/null' || true
    fi
}

setup_gpg_forward() {
    if [ "${DEV_HAS_SOCAT:-false}" != "true" ]; then
        echo "warning: socat not found on host — GPG agent forwarding disabled (apt install socat)" >&2
        return 0
    fi

    # agent-extra-socket is specifically designed for forwarding to remote machines
    local host_gpg_socket
    host_gpg_socket="$(gpgconf --list-dirs agent-extra-socket 2>/dev/null)" || true
    if [ -z "$host_gpg_socket" ] || [ ! -S "$host_gpg_socket" ]; then
        echo "warning: no GPG agent extra socket detected — SOPS PGP decryption won't work inside the container" >&2
        echo "  (ensure gpg-agent is running: gpgconf --launch gpg-agent)" >&2
        return 0
    fi

    # Warm gpg-agent's internal passphrase cache. The extra socket used for
    # forwarding passes --no-allow-external-cache to pinentry, so the
    # passphrase must be in gpg-agent's own cache — an external keyring
    # (GNOME Keyring, macOS Keychain, etc.) won't be consulted.
    #
    # Two operations are needed: a sign (caches the signing key) and an
    # encrypt-to-self + decrypt (caches the encryption subkey). SOPS
    # uses decryption, so skipping the second leaves the container unable
    # to decrypt secrets.
    if ! echo | gpg -so /dev/null 2>/dev/null; then
        echo "note: could not pre-cache GPG signing passphrase — you may be prompted inside the container" >&2
    fi
    local default_key
    default_key="$(gpg --list-secret-keys --with-colons 2>/dev/null | awk -F: '/^sec/{print $5; exit}')"
    if [ -n "$default_key" ]; then
        if ! echo "warmup" | gpg -e -r "$default_key" 2>/dev/null | gpg -d >/dev/null 2>&1; then
            echo "note: could not pre-cache GPG decryption passphrase — SOPS may prompt inside the container" >&2
        fi
    fi

    stop_gpg_forward

    local port
    port="$(gpg_forward_port)"

    local bind_addr
    if [ "$(uname -s)" = "Linux" ] && ! grep -qiF microsoft /proc/version 2>/dev/null; then
        bind_addr="$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)" || true
        if [ -z "$bind_addr" ]; then
            echo "error: could not determine Docker bridge gateway IP — is the bridge network enabled?" >&2
            return 1
        fi
    else
        bind_addr="127.0.0.1"
    fi

    socat "TCP-LISTEN:${port},bind=${bind_addr},reuseaddr,fork" \
          "UNIX-CONNECT:${host_gpg_socket}" &
    local host_pid=$!
    echo "$host_pid" > "$DEV_GPG_FORWARD_PIDFILE"

    local i
    for i in $(seq 1 50); do
        sleep 0.05
        kill -0 "$host_pid" 2>/dev/null || break
    done
    if ! kill -0 "$host_pid" 2>/dev/null; then
        echo "error: host-side GPG forwarder failed to start (port $port may be in use)" >&2
        rm -f "$DEV_GPG_FORWARD_PIDFILE"
        return 1
    fi

    # Kill any container-local gpg-agent so it doesn't compete for the socket
    dc_exec gpgconf --kill gpg-agent 2>/dev/null || true

    dc exec -d app socat \
        UNIX-LISTEN:/run/gpg-agent/S.gpg-agent,fork,unlink-early,mode=600 \
        TCP:host.docker.internal:${port}

    # Point container gpg at the forwarded socket
    dc_exec bash -c 'mkdir -p ~/.gnupg && chmod 700 ~/.gnupg && ln -sf /run/gpg-agent/S.gpg-agent ~/.gnupg/S.gpg-agent'

    # GPG needs public keys in the local keyring to discover which secret
    # keys the forwarded agent holds. Without this, gpg --list-secret-keys
    # returns nothing and decryption fails with "No secret key".
    gpg --export 2>/dev/null | dc exec -T app gpg --import 2>/dev/null || true

    for i in $(seq 1 30); do
        if dc_exec gpg --list-secret-keys 2>/dev/null | grep -q '^sec'; then
            return 0
        fi
        sleep 0.1
    done
    echo "warning: GPG agent bridge started but no secret keys visible inside the container" >&2
    echo "  (is the private key loaded on the host? check: gpg --list-secret-keys)" >&2
    return 0
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
        # docker cp preserves host UID; explicit chown makes us robust to
        # base images that don't honor USER_UID build args.
        dc exec -u root app chown vscode:vscode /home/vscode/.config/gh/hosts.yml
    fi
}

setup_claude_credentials() {
    # Host-side prep for the credentials bind-mount declared in compose.yml.
    # Two responsibilities:
    #   1. Ensure ~/.claude/credentials/ exists so the bind-mount source is
    #      present (otherwise Docker creates it root-owned).
    #   2. Migrate any legacy ~/.claude/.credentials.json into credentials/
    #      and leave a host-side symlink so host claude-code follows the
    #      same shared file.
    # The container-side ~/.claude/.credentials.json symlink is image-baked
    # in the Dockerfile and not managed here.
    local cred_dir="$HOME/.claude/credentials"
    local cred_file="$HOME/.claude/.credentials.json"

    mkdir -p "$cred_dir"

    if [ -f "$cred_file" ] && [ ! -L "$cred_file" ]; then
        mv "$cred_file" "$cred_dir/.credentials.json"
        ln -s credentials/.credentials.json "$cred_file"
    fi
}

setup_claude() {
    local host_project_key container_project_key
    host_project_key="$(echo "$DEV_WORKSPACE" | sed 's|/|-|g')"
    container_project_key="$(echo "$DEV_CONTAINER_WORKSPACE" | sed 's|/|-|g')"

    # Named volume root comes up root-owned; fix so vscode can write.
    # Top-level only — recursing would descend into the credentials/ bind
    # mount and rewrite ownership on host files. Image-baked contents
    # (.credentials.json symlink, subdirs) are already vscode-owned.
    dc exec -u root app chown vscode:vscode /home/vscode/.claude

    dc exec \
        -e DEV_CONTAINER_WORKSPACE="$DEV_CONTAINER_WORKSPACE" \
        -e DEV_HOST_PROJECT_KEY="$host_project_key" \
        -e DEV_CONTAINER_PROJECT_KEY="$container_project_key" \
        app python3 /usr/local/bin/setup-claude
}
