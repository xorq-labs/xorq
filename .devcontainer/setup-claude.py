#!/usr/bin/env python3
"""Set up Claude Code config inside the dev container.

Copies host baseline (credentials, permissions, memory) into the container's
isolated ~/.claude, installs a PreToolUse audit hook, and symlinks sessions
to the workspace for host log capture.

Expected environment variables (set by dev/devcontainer):
    DEV_CONTAINER_WORKSPACE  — container workspace path (e.g. /workspaces/src)
    DEV_HOST_PROJECT_KEY     — mangled host workspace path (e.g. -home-dan-repos-github-xorq)
    DEV_CONTAINER_PROJECT_KEY — mangled container workspace path (e.g. -workspaces-src)
"""

import json
import os
import shutil
import sys
from pathlib import Path


HOST = Path("/home/vscode/.claude-host")
HOME = Path("/home/vscode/.claude")
HOST_PREFS = Path("/home/vscode/.claude-host.json")
CONTAINER_PREFS = Path("/home/vscode/.claude.json")

REQUIRED_VARS = (
    "DEV_CONTAINER_WORKSPACE",
    "DEV_HOST_PROJECT_KEY",
    "DEV_CONTAINER_PROJECT_KEY",
)


def copy_credentials():
    src = HOST / ".credentials.json"
    if src.exists():
        dst = HOME / ".credentials.json"
        shutil.copy2(src, dst)
        dst.chmod(0o600)


def copy_global_instructions():
    src = HOST / "CLAUDE.md"
    if src.exists():
        shutil.copy2(src, HOME / "CLAUDE.md")


def copy_user_prefs(workspace):
    prefs = {}
    if HOST_PREFS.exists():
        with open(HOST_PREFS) as f:
            prefs = json.load(f)

    projects = prefs.setdefault("projects", {})
    ws_key = str(workspace)
    projects.setdefault(ws_key, {})
    projects[ws_key]["hasTrustDialogAccepted"] = True

    with open(CONTAINER_PREFS, "w") as f:
        json.dump(prefs, f, indent=2)


def setup_settings(workspace, host_project_key):
    host_settings = {}
    src = HOST / "settings.json"
    if src.exists():
        with open(src) as f:
            host_settings = json.load(f)

    skipped_hooks = 0
    for hook_list in host_settings.get("hooks", {}).values():
        for matcher_group in hook_list:
            skipped_hooks += len(matcher_group.get("hooks", []))

    host_project_dir = HOST / "projects" / host_project_key
    if host_project_dir.is_dir():
        for name in ("settings.json", "settings.local.json"):
            src = host_project_dir / name
            if not src.exists():
                continue
            with open(src) as f:
                proj = json.load(f)
            for hook_list in proj.get("hooks", {}).values():
                for matcher_group in hook_list:
                    skipped_hooks += len(matcher_group.get("hooks", []))

    if skipped_hooks:
        print(f"note: skipping {skipped_hooks} host hook(s) (reference host paths)")

    audit_log = workspace / ".claude" / "container-audit" / "audit.jsonl"

    audit_cmd = f"python3 /usr/local/bin/audit-hook {audit_log}"

    container_settings = {
        "permissions": host_settings.get("permissions", {}),
        "skipDangerousModePermissionPrompt": True,
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "",
                    "hooks": [
                        {"type": "command", "command": audit_cmd},
                    ],
                }
            ],
        },
    }

    with open(HOME / "settings.json", "w") as f:
        json.dump(container_settings, f, indent=2)


def setup_project_settings(host_project_key, container_project_key):
    host_project_dir = HOST / "projects" / host_project_key
    container_project_dir = HOME / "projects" / container_project_key

    if container_project_dir.is_symlink():
        container_project_dir.unlink()

    container_project_dir.mkdir(parents=True, exist_ok=True)

    if not host_project_dir.is_dir():
        return

    for name in ("settings.json", "settings.local.json"):
        src = host_project_dir / name
        if not src.exists():
            continue
        with open(src) as f:
            proj = json.load(f)
        container_proj = {"permissions": proj.get("permissions", {})}
        with open(container_project_dir / name, "w") as f:
            json.dump(container_proj, f, indent=2)

    host_memory = host_project_dir / "memory"
    container_memory = container_project_dir / "memory"
    if host_memory.is_dir() and not container_memory.exists():
        shutil.copytree(host_memory, container_memory)


def setup_sessions(workspace):
    sessions_target = workspace / ".claude" / "container-sessions"
    sessions_target.mkdir(parents=True, exist_ok=True)

    sessions_link = HOME / "sessions"
    if sessions_link.is_dir() and not sessions_link.is_symlink():
        shutil.rmtree(sessions_link)
    elif sessions_link.is_symlink() or sessions_link.exists():
        sessions_link.unlink()
    sessions_link.symlink_to(sessions_target)


def setup_audit(workspace):
    (workspace / ".claude" / "container-audit").mkdir(parents=True, exist_ok=True)


def main():
    missing = [v for v in REQUIRED_VARS if v not in os.environ]
    if missing:
        print(
            f"error: missing environment variables: {', '.join(missing)}",
            file=sys.stderr,
        )
        print(
            "This script should be called via dev/devcontainer, not directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    workspace = Path(os.environ["DEV_CONTAINER_WORKSPACE"])
    host_project_key = os.environ["DEV_HOST_PROJECT_KEY"]
    container_project_key = os.environ["DEV_CONTAINER_PROJECT_KEY"]

    HOME.mkdir(parents=True, exist_ok=True)

    copy_credentials()
    copy_global_instructions()
    copy_user_prefs(workspace)
    setup_settings(workspace, host_project_key)
    setup_project_settings(host_project_key, container_project_key)
    setup_sessions(workspace)
    setup_audit(workspace)


if __name__ == "__main__":
    main()
