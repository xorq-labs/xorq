#!/usr/bin/env python3
"""Analyze the container audit log and report tool usage."""

import json
import sys
from collections import Counter
from pathlib import Path


def load_two_word_prefixes():
    # Format rules mirror lib/list-file.sh's read_list — keep in sync.
    path = Path(__file__).parent / "project" / "audit-prefixes.txt"
    prefixes = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                prefixes.add(line)
    return prefixes


def main():
    audit_log = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--summary"

    host_settings_path = Path.home() / ".claude" / "settings.json"
    baseline = set()
    try:
        with open(host_settings_path) as f:
            for perm in json.load(f).get("permissions", {}).get("allow", []):
                baseline.add(perm)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    two_word_prefixes = load_two_word_prefixes()

    tool_counts = Counter()
    bash_prefixes = Counter()
    patterns = set()

    with open(audit_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool = entry.get("tool", "")
            inp = entry.get("input", {})
            tool_counts[tool] += 1
            if tool == "Bash":
                cmd = inp.get("command", "")
                parts = cmd.split()
                if not parts:
                    continue
                prefix = parts[0]
                if prefix in two_word_prefixes and len(parts) > 1:
                    prefix = f"{parts[0]} {parts[1]}"
                bash_prefixes[prefix] += 1
                patterns.add(f"Bash({prefix}:*)")
            else:
                patterns.add(tool)

    if mode == "--summary":
        print("Tool usage:")
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count}")
        if bash_prefixes:
            print("\nBash command prefixes:")
            for cmd, count in bash_prefixes.most_common():
                print(f"  {cmd}: {count}")
        new = sorted(patterns - baseline)
        if new:
            print(f"\nPermissions not in host baseline ({len(new)}):")
            for p in new:
                print(f"  {p}")
        else:
            print("\nAll observed patterns covered by host baseline.")
    elif mode == "--new":
        for p in sorted(patterns - baseline):
            print(p)
    elif mode == "--all":
        for p in sorted(patterns):
            print(p)
    else:
        print(f"Unknown audit mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
