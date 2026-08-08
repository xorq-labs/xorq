"""Give the in-flight ADR the number of this branch's pull request.

    python3 scripts/adr_rename.py            # one ADR in flight
    python3 scripts/adr_rename.py <slug>     # pick one of several

Renames ``docs/adr/XXXX-<slug>.md`` to ``docs/adr/<pr>-<slug>.md``, fixes the
heading, repoints citations at the new number, and re-runs the guard.

Prose citations are a readability pass: an ADR is citable as ``ADR-<slug>`` and
as ``ADR-<number>``, both resolve forever, so one the sweep misses stays valid.
Relative links to the placeholder filename are not optional in the same way --
the file has moved, so those are rewritten exactly.

Stdlib only, matching scripts/adr_check.py.
"""

from __future__ import annotations

import json
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


def pull_request() -> tuple[str, str] | None:
    """This branch's pull request number and base branch.

    The base matters: a stacked pull request is based on its parent branch, not
    on `main`, and diffing against the wrong base attributes every ADR the
    parent added to this pull request as well.
    """
    result = subprocess.run(
        ["gh", "pr", "view", "--json", "number,baseRefName"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        sys.stderr.write("no pull request found for this branch; open one first\n")
        return None
    try:
        payload = json.loads(result.stdout)
        return (str(payload["number"]), payload["baseRefName"])
    except (ValueError, KeyError):
        sys.stderr.write(f"could not read the pull request: {result.stdout!r}\n")
        return None


def sweep_references(slug: str, number: str) -> int:
    """Point every citation of an ADR at its new number.

    Two forms move together: the prose citation ``ADR-<slug>`` and any relative
    link to the placeholder filename. Missing the second would leave a link to
    a file the rename just moved, which the guard reports as a dead link -- so
    the sweep is best-effort about prose but must be exact about paths.
    """
    # The lookahead stops a slug that prefixes a longer one from matching it:
    # sweeping `tee-node` must leave `ADR-tee-node-deferred-writes` alone.
    rewrites = (
        (re.compile(rf"ADR-{re.escape(slug)}(?![a-z0-9-])"), f"ADR-{number}"),
        (
            re.compile(rf"(?<=\()\s*{PLACEHOLDER}-{re.escape(slug)}\.md(?=[)#])"),
            f"{number}-{slug}.md",
        ),
    )
    found = subprocess.run(
        [
            "git",
            "grep",
            "--untracked",
            "-lIe",
            f"ADR-{slug}",
            "-e",
            f"{PLACEHOLDER}-{slug}.md",
        ],
        capture_output=True,
        text=True,
    )
    changed = 0
    for line in found.stdout.splitlines():
        if not line:
            continue
        path = Path(line)
        text = path.read_text(encoding="utf-8")
        rewritten = text
        for pattern, replacement in rewrites:
            rewritten = pattern.sub(replacement, rewritten)
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

    found = pull_request()
    if found is None:
        return 1
    number, base = found

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

    # Located next to this file rather than by relative path, so the guard is
    # found no matter where the caller invoked us from.
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().with_name("adr_check.py")),
            "--base",
            base,
            "--pr",
            number,
        ],
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
