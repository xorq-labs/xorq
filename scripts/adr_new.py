"""Scaffold a new ADR from the template.

    python3 scripts/adr_new.py <slug>

Writes ``docs/adr/XXXX-<slug>.md``. The XXXX placeholder stays until the pull
request exists; ``scripts/adr_rename.py`` replaces it with the pull request
number. Until then the ADR is citable as ``ADR-<slug>``.

Stdlib only, matching scripts/adr_check.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ADR_DIR = Path("docs/adr")
TEMPLATE = ADR_DIR / "template.md"
PLACEHOLDER = "XXXX"

# Must agree with FILENAME_RE's slug group in adr_check.py, including the
# letter-initial rule that keeps a slug from parsing as a number.
SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {sys.argv[0]} <slug>\n")
        return 2
    slug = sys.argv[1]

    if not SLUG_RE.match(slug):
        sys.stderr.write(
            f"{slug!r} is not a valid slug: lowercase words joined by single "
            "hyphens and starting with a letter, for example "
            "content-store-capability-and-binding\n"
        )
        return 1

    if not TEMPLATE.is_file():
        sys.stderr.write(f"{TEMPLATE} not found; run from the repository root\n")
        return 2

    dest = ADR_DIR / f"{PLACEHOLDER}-{slug}.md"
    if dest.exists():
        sys.stderr.write(f"{dest} already exists\n")
        return 1

    dest.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    sys.stdout.write(
        f"created {dest}\n"
        f"cite it as ADR-{slug} until it has a number\n"
        "next: write it, open the pull request, then run "
        "`python3 scripts/adr_rename.py`\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
