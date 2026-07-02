#!/usr/bin/env bash
set -euo pipefail

# Freeze a build's caches with `xorq pin` (rewrites each cache into a direct
# read of its file) and thaw them with `xorq unpin`. --relocate-reads bundles
# those files into the build for a self-contained, portable artifact.

script="${1:?Usage: ./pin_unpin.sh <script.py> [-e expr_name]}"
shift || true

expr_name="expr"
cache_dir="./pin-unpin-cache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e) expr_name="$2"; shift 2 ;;
        --cache-dir) cache_dir="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "==> Building '${expr_name}' from ${script}"
build_path=$(xorq build "$script" -e "$expr_name" --cache-dir "$cache_dir")
echo "    build: ${build_path}"

# Pin requires materialized caches; run once to populate them (or use `pin -e`).
echo "==> Running build to materialize caches"
# `head` closes the pipe early, so under `set -o pipefail` xorq's SIGPIPE (141)
# would fail the script; `|| true` keeps the demo going. The caches are still
# fully written -- materialization is a side effect of the run, not of how many
# rows are read back.
xorq run "$build_path" -f json -o - --cache-dir "$cache_dir" | head -n 5 || true

echo "==> Pinning (freeze caches into direct reads, bundle them in)"
# --relocate-reads is on by default; shown here for clarity.
pinned_path=$(xorq pin "$build_path" --cache-dir "$cache_dir" --relocate-reads)
echo "    pinned: ${pinned_path}"
# Pinning intentionally changes the build hash: a distinct build, not an edit.

echo "==> Unpinning (thaw frozen reads back into recomputable caches)"
unpinned_path=$(xorq unpin "$pinned_path" --cache-dir "$cache_dir")
echo "    unpinned: ${unpinned_path}"
# Unpin reverses pin's cache transform. The hash only returns to the original
# build's when reads are not relocated on either leg (relocate rewrites read
# identity, so a relocated build is not load+rebuild hash-stable); see
# `xorq pin/unpin --no-relocate-reads` for the hash-stable round-trip.
