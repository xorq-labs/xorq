"""Print an index of the ADRs in docs/adr, generated from the files themselves.

    python3 scripts/adr_index.py

A workflow script, in the sense scripts/README.md means: run by hand, by a
person who reads the output. It is emphatically not a guard. Nothing in CI
invokes it, and nothing should -- see below.

Deliberately printed, never committed: `docs/adr/README.md`'s "Why there is no
index file" section is the argument, and generating the table rather than
writing it does not answer it. Nothing depends on this output, which is what
makes it safe to have.

It sorts correctly, which `ls` will not once pull request numbers reach five
digits: `10000-x.md` sorts before `2129-x.md` lexically. Numbers are not
zero-padded -- an ADR's number is its pull request's number, written the way
the forge writes it -- so ordering belongs in the tool that reads them.

Stdlib only, matching scripts/adr_check.py.
"""

from __future__ import annotations

import re
import sys

# The guard owns what an ADR is called and where they live; take those from it
# rather than a copy, the way adr_rename.py does. A sibling import works because
# python puts this script's directory on sys.path.
from adr_check import ADR_DIR, HEADING_RE, adr_files, parse_name


# `- **Status:** Accepted`, per docs/adr/template.md.
STATUS_RE = re.compile(r"^-\s+\*\*Status:\*\*\s*(?P<status>.+?)\s*$", re.MULTILINE)

# A status may cite the ADR that superseded it as a link. The index shows text,
# not links: the target is one click away in the ADR itself, and a table full of
# relative links is harder to read than the sentence it came from.
MD_LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\([^)]*\)")


def title_of(text: str) -> str:
    """The heading's title, with the `ADR-NNNN:` prefix removed."""
    match = HEADING_RE.search(text)
    if match is None:
        return ""
    line = text[match.start() : text.find("\n", match.start())]
    _, _, title = line.partition(":")
    return title.strip()


def status_of(text: str) -> str:
    match = STATUS_RE.search(text)
    if match is None:
        return "—"
    return MD_LINK_RE.sub(r"\g<text>", match.group("status")).strip()


def entries() -> list[tuple[int | None, str, str, str, str]]:
    """One tuple per ADR: (number, label, slug, title, status).

    Sorted by number, with any still-unnumbered ADR last -- it has no place in
    a numeric sequence yet, and putting it first would imply it comes before
    ADR-0002.
    """
    found = []
    for path in adr_files():
        try:
            number, slug, raw = parse_name(path.name)
        except ValueError:
            continue  # malformed; adr_check.py reports it precisely
        text = path.read_text(encoding="utf-8")
        found.append((number, raw, slug, title_of(text), status_of(text)))
    return sorted(found, key=lambda row: (row[0] is None, row[0] or 0, row[2]))


def render(rows: list[tuple[int | None, str, str, str, str]]) -> str:
    lines = ["| ADR | Title | Status |", "|-----|-------|--------|"]
    for _number, raw, slug, title, status in rows:
        link = f"[{raw}]({raw}-{slug}.md)"
        lines.append(f"| {link} | {title} | {status} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write(f"usage: {sys.argv[0]}\n")
        return 2
    if not ADR_DIR.is_dir():
        sys.stderr.write(f"{ADR_DIR} not found; run from the repository root\n")
        return 2
    rows = entries()
    if not rows:
        sys.stderr.write(f"no ADRs in {ADR_DIR}\n")
        return 1
    sys.stdout.write(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
