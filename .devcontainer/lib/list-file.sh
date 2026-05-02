# Shared parser for the data-driven list files under .devcontainer/project/
# (worktree-symlinks.txt, worktree-copies.txt, external-volumes.txt,
# audit-prefixes.txt, ...).
#
# Format:
#   - one entry per line
#   - everything from the first '#' onward is treated as a comment
#   - leading and trailing whitespace is trimmed
#   - blank lines (after stripping) are skipped
#   - a missing file is treated as empty
#   - '#' has no escape — entries containing a literal '#' aren't expressible
#
# Python callers (audit-report.py) implement the same rules inline; keep
# them in sync with this function.

read_list() {
    local file="$1"
    [ -f "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "$line" ] && continue
        printf '%s\n' "$line"
    done < "$file"
}
