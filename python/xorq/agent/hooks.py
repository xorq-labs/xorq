from __future__ import annotations

import stat
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable


LAND_SCRIPT = (
    textwrap.dedent(
        """\
    #!/usr/bin/env bash
    # Enforce the xorq landing checklist before critical git actions.

    set -euo pipefail

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    cd "${REPO_ROOT}"

    if ! command -v xorq >/dev/null 2>&1; then
        echo "[xorq-land] xorq CLI is not available in PATH." >&2
        exit 1
    fi

    export HOME="${REPO_ROOT}"

    echo "[xorq-land] Running \\`xorq agent land\\` to show the required checklist..."
    if ! xorq agent land --limit 5; then
        echo "[xorq-land] The landing checklist failed. Resolve the issues above before continuing." >&2
        exit 1
    fi

    if [ -n "${XORQ_HOOKS_ASSUME_YES:-}" ] || { [ -n "${CI:-}" ] && [ "${CI}" != "false" ]; }; then
        echo "[xorq-land] AUTO mode enabled (XORQ_HOOKS_ASSUME_YES/CI). Skipping interactive confirmation."
        exit 0
    fi

    if [ -t 0 ]; then
        read -r -p "[xorq-land] Type 'landed' to confirm build → catalog → run → push is complete: " response
        if [ "${response}" != "landed" ]; then
            echo "[xorq-land] Landing checklist not acknowledged. Aborting." >&2
            exit 1
        fi
        echo "[xorq-land] Landing checklist acknowledged."
    else
        echo "[xorq-land] Non-interactive shell; set XORQ_HOOKS_ASSUME_YES=1 to bypass confirmation." >&2
        exit 1
    fi
    """
    ).strip()
    + "\n"
)


PRE_COMMIT_HOOK = (
    textwrap.dedent(
        """\
    #!/usr/bin/env bash
    # Git pre-commit hook wrapper for xorq landing checklist.

    set -euo pipefail

    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    HOOK_HELPER="${REPO_ROOT}/scripts/xorq-land.sh"

    if [ ! -x "${HOOK_HELPER}" ]; then
        echo "[xorq-hook] Helper ${HOOK_HELPER} is missing or not executable." >&2
        exit 1
    fi

    "${HOOK_HELPER}"
    """
    ).strip()
    + "\n"
)


POST_MERGE_HOOK = (
    textwrap.dedent(
        """\
    #!/usr/bin/env bash
    # Git post-merge hook that refreshes xorq workflow guidance.

    set -euo pipefail

    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    cd "${REPO_ROOT}"

    if ! command -v xorq >/dev/null 2>&1; then
        exit 0
    fi

    if [ ! -d ".xorq" ]; then
        exit 0
    fi

    export HOME="${REPO_ROOT}"
    OUTPUT_PATH=".xorq/LAST_PRIME.md"

    {
        echo "[xorq-post-merge] Updating PRIME guidance after merge..."
        xorq agent prime > "${OUTPUT_PATH}"
        echo "[xorq-post-merge] Latest workflow context saved to ${OUTPUT_PATH}."
        echo "[xorq-post-merge] Run 'xorq agent prime' for the interactive view."
    } >&2
    """
    ).strip()
    + "\n"
)


INSTALL_SCRIPT = (
    textwrap.dedent(
        """\
    #!/usr/bin/env bash
    # Copy xorq hook templates into .git/hooks

    set -euo pipefail

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    HOOK_SOURCE="${REPO_ROOT}/.xorq/hooks"
    HOOK_DEST="${REPO_ROOT}/.git/hooks"

    if [ ! -d "${HOOK_DEST}" ]; then
        echo "[xorq-hooks] ${HOOK_DEST} is missing. Run this from inside a git repository." >&2
        exit 1
    fi

    if [ ! -d "${HOOK_SOURCE}" ]; then
        echo "[xorq-hooks] No hooks found in ${HOOK_SOURCE}." >&2
        exit 1
    fi

    copied=0
    for hook in "${HOOK_SOURCE}"/*; do
        name="$(basename "${hook}")"
        cp "${hook}" "${HOOK_DEST}/${name}"
        chmod +x "${HOOK_DEST}/${name}"
        echo "[xorq-hooks] Installed ${name} hook."
        copied=$((copied + 1))
    done

    if [ "${copied}" -eq 0 ]; then
        echo "[xorq-hooks] No hooks were installed." >&2
        exit 1
    fi

    echo "[xorq-hooks] Done. Git will now enforce the xorq workflow via these hooks."
    """
    ).strip()
    + "\n"
)


PRIME_GUIDANCE = (
    textwrap.dedent(
        """\
    # xorq Project-Specific Guidance

    - Always start a session with `xorq agent onboard` (for the full checklist) and `xorq agent prime` (for dynamic context). This file overrides the default PRIME output.
    - Install the repo hooks once per clone:
      ```bash
      scripts/install-xorq-hooks.sh
      ```
      This copies the template hooks under `.xorq/hooks/` into `.git/hooks/`.
    - The **pre-commit hook** runs `scripts/xorq-land.sh`, which:
      1. Executes `xorq agent land` so you see the landing checklist.
      2. Requires you to acknowledge that build → catalog → run → push is complete (`type 'landed'`).
    - The **post-merge hook** refreshes `.xorq/LAST_PRIME.md` by calling `xorq agent prime`. Open that file—or rerun `xorq agent prime`—after every pull/merge to rehydrate the workflow context.
    - To bypass the interactive confirmation (CI or automation), export `XORQ_HOOKS_ASSUME_YES=1`.
    - Hooks expect the xorq CLI to run from the repo root. If you relocate the working copy, rerun the installer.
    """
    ).strip()
    + "\n"
)


def _write_if_missing(
    path: Path, content: str, executable: bool = False
) -> Path | None:
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
    return path


def ensure_agent_hook_files(project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    created: list[Path] = []

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    artefacts: Iterable[tuple[Path, str, bool]] = (
        (scripts_dir / "xorq-land.sh", LAND_SCRIPT, True),
        (scripts_dir / "install-xorq-hooks.sh", INSTALL_SCRIPT, True),
        (root / ".xorq" / "hooks" / "pre-commit", PRE_COMMIT_HOOK, True),
        (root / ".xorq" / "hooks" / "post-merge", POST_MERGE_HOOK, True),
        (root / ".xorq" / "PRIME.md", PRIME_GUIDANCE, False),
    )
    for path, content, executable in artefacts:
        maybe_created = _write_if_missing(path, content, executable)
        if maybe_created is not None:
            created.append(maybe_created)

    # Always ensure executability for scripts.
    for script_path in (
        scripts_dir / "xorq-land.sh",
        scripts_dir / "install-xorq-hooks.sh",
        root / ".xorq" / "hooks" / "pre-commit",
        root / ".xorq" / "hooks" / "post-merge",
    ):
        if script_path.exists():
            script_path.chmod(
                stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )

    # Auto-install hooks if possible
    hook_installer = scripts_dir / "install-xorq-hooks.sh"
    git_hooks_dir = root / ".git" / "hooks"
    if hook_installer.exists():
        if git_hooks_dir.exists():
            try:
                subprocess.run(
                    [str(hook_installer)],
                    cwd=root,
                    check=True,
                )
            except Exception as exc:  # pragma: no cover - best-effort log
                print(f"[xorq-hooks] Failed to install hooks automatically: {exc}")
        else:
            print("[xorq-hooks] Skipping hook install (no .git/hooks directory found).")

    return created
