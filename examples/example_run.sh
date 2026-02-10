#!/usr/bin/env bash
set -euo pipefail

# This is a utility script to build an expression then immediately run it. Normally you would run `xorq build` and `xorq run` separately

script="${1:?Usage: ./example_run.sh <script.py> [-e expr_name] [-f format] [-o output]}"
shift

expr_name="expr"
format="json"
output="-"
head_lines="10"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e) expr_name="$2"; shift 2 ;;
        -f) format="$2"; shift 2 ;;
        -o) output="$2"; shift 2 ;;
        -n) head_lines="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "==> Building '${expr_name}' from ${script}"
build_path=$(xorq build "$script" -e "$expr_name")

echo "==> Running build at ${build_path}"
if [[ "$format" == "json" && "$output" == "-" && -n "$head_lines" ]]; then
    #python will error with a BrokenPipe Error confusing the user, so we just send stderr to /dev/null
    #its a bit of a hack, but gives a good default user experience
    xorq run "$build_path" -f "$format" -o "$output" 2>/dev/null | head -n "$head_lines"
else
    xorq run "$build_path" -f "$format" -o "$output"
fi
