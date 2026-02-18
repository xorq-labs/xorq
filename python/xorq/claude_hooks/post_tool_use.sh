#!/usr/bin/env bash
# PostToolUse hook - logs failures to a JSONL file for the stop hook to analyze.
# Rewritten from Python to bash+jq for speed (avoids ~100ms Python startup per tool call).
set -euo pipefail

INPUT=$(cat)

ERROR=$(printf '%s' "$INPUT" | jq -r '.error // empty')
[ -z "$ERROR" ] && exit 0

TRANSCRIPT_PATH=$(printf '%s' "$INPUT" | jq -r '.transcript_path // empty')
[ -z "$TRANSCRIPT_PATH" ] && exit 0

SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // "unknown"')
SESSIONS_DIR=$(dirname "$TRANSCRIPT_PATH")
FAILURES_PATH="${SESSIONS_DIR}/${SESSION_ID}.failures.jsonl"

TIMESTAMP=$(date +%s)
TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // ""')
TOOL_INPUT=$(printf '%s' "$INPUT" | jq -c '.tool_input // {}')
ERROR_SUMMARY=$(printf '%s' "$INPUT" | jq -r '(.error // "")[:500]')

printf '%s\n' "$(jq -nc \
  --arg ts "$TIMESTAMP" \
  --arg tn "$TOOL_NAME" \
  --argjson ti "$TOOL_INPUT" \
  --arg es "$ERROR_SUMMARY" \
  '{timestamp: ($ts | tonumber), tool_name: $tn, tool_input: $ti, error_summary: $es}')" \
  >> "$FAILURES_PATH"
