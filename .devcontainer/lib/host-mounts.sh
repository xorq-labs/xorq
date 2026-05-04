# Reads host-mount list files and generates a docker-compose override that
# bind-mounts host paths into the container.
#
# Mount-list format (same parser as list-file.sh):
#   <host-path>:<container-path>[:<options>]
#   ~ at the start of host-path expands to $HOME.
#   Host paths that don't exist are skipped with a warning.
#
# Two tiers:
#   host-mounts.txt        — committed, project-wide
#   host-mounts.local.txt  — gitignored, per-developer
#
# Sourcer must provide: DEV_MAIN_TREE, DEV_CONTAINER_NAME, read_list()

generate_host_mounts_override() {
    local project_file="$DEV_MAIN_TREE/.devcontainer/project/host-mounts.txt"
    local local_file="$DEV_MAIN_TREE/.devcontainer/project/host-mounts.local.txt"
    local out="/tmp/devcontainer-host-mounts-${DEV_CONTAINER_NAME}.yml"
    local mounts=()

    local line host_path
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        line="${line/#\~/$HOME}"
        host_path="${line%%:*}"
        if [ ! -e "$host_path" ]; then
            echo "warning: host-mounts: skipping (host path does not exist): $line" >&2
            continue
        fi
        mounts+=("$line")
    done < <(read_list "$project_file"; read_list "$local_file")

    if [ ${#mounts[@]} -eq 0 ]; then
        rm -f "$out"
        DEV_HOST_MOUNTS_OVERRIDE=""
        return 0
    fi

    printf 'services:\n  app:\n    volumes:\n' > "$out"
    local m
    for m in "${mounts[@]}"; do
        printf '      - "%s"\n' "$m" >> "$out"
    done

    DEV_HOST_MOUNTS_OVERRIDE="$out"
}
