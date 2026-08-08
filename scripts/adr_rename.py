"""Give the in-flight ADR the number of this branch's pull request.

    python3 scripts/adr_rename.py            # one ADR in flight
    python3 scripts/adr_rename.py <slug>     # pick one of several

Renames ``docs/adr/XXXX-<slug>.md`` to ``docs/adr/<pr>-<slug>.md``, fixes the
heading, rewrites named references to the numeric form, and re-runs the guard.

The reference sweep is a readability pass, never a correctness one: an ADR is
citable as ``ADR-<slug>`` and as ``ADR-<number>`` and both resolve forever, so
a reference the sweep misses stays valid rather than breaking the build.

Stdlib only, matching scripts/adr_check.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ADR_DIR = Path("docs/adr")
PLACEHOLDER = "XXXX"


def unnumbered() -> list[Path]:
    return sorted(ADR_DIR.glob(f"{PLACEHOLDER}-*.md"))


def pick_source(slug: str | None) -> Path | None:
    """The ADR to number, or None after reporting why it is ambiguous."""
    if slug is not None:
        src = ADR_DIR / f"{PLACEHOLDER}-{slug}.md"
        if not src.exists():
            sys.stderr.write(f"{src} does not exist\n")
            return None
        return src

    candidates = unnumbered()
    if not candidates:
        sys.stderr.write(f"no {ADR_DIR}/{PLACEHOLDER}-*.md to rename\n")
        return None
    if len(candidates) > 1:
        sys.stderr.write("more than one unnumbered ADR:\n")
        for path in candidates:
            sys.stderr.write(f"  {path}\n")
        sys.stderr.write(
            f"pass the one this pull request adds: python3 {sys.argv[0]} <slug>\n"
        )
        return None
    return candidates[0]


def pull_request_number() -> str | None:
    result = subprocess.run(
        ["gh", "pr", "view", "--json", "number", "--jq", ".number"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        sys.stderr.write("no pull request found for this branch; open one first\n")
        return None
    return result.stdout.strip()


def sweep_references(slug: str, number: str) -> int:
    """Rewrite ADR-<slug> to ADR-<number> wherever it appears."""
    # The lookahead stops a slug that prefixes a longer one from matching it:
    # sweeping `tee-node` must leave `ADR-tee-node-deferred-writes` alone.
    pattern = re.compile(rf"ADR-{re.escape(slug)}(?![a-z0-9-])")
    found = subprocess.run(
        ["git", "grep", "--untracked", "-lIF", f"ADR-{slug}"],
        capture_output=True,
        text=True,
    )
    changed = 0
    for line in found.stdout.splitlines():
        if not line:
            continue
        path = Path(line)
        text = path.read_text(encoding="utf-8")
        rewritten = pattern.sub(f"ADR-{number}", text)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    if len(sys.argv) > 2:
        sys.stderr.write(f"usage: {sys.argv[0]} [slug]\n")
        return 2
    if not ADR_DIR.is_dir():
        sys.stderr.write(f"{ADR_DIR} not found; run from the repository root\n")
        return 2

    src = pick_source(sys.argv[1] if len(sys.argv) == 2 else None)
    if src is None:
        return 1

    number = pull_request_number()
    if number is None:
        return 1

    slug = src.name[len(PLACEHOLDER) + 1 : -len(".md")]
    dest = ADR_DIR / f"{number}-{slug}.md"

    subprocess.run(["git", "mv", str(src), str(dest)], check=True)
    dest.write_text(
        dest.read_text(encoding="utf-8").replace(
            f"# ADR-{PLACEHOLDER}:", f"# ADR-{number}:", 1
        ),
        encoding="utf-8",
    )
    sys.stdout.write(f"renamed to {dest}\n")

    changed = sweep_references(slug, number)
    sys.stdout.write(
        f"rewrote ADR-{slug} to ADR-{number} in {changed} file(s)\n"
        if changed
        else f"no named references to ADR-{slug} to rewrite\n"
    )

    return subprocess.run(
        [
            sys.executable,
            "scripts/adr_check.py",
            "--base",
            "main",
            "--pr",
            number,
        ],
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
