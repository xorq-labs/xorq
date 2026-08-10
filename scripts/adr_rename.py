"""Give the in-flight ADR the number of this branch's pull request.

    python3 scripts/adr_rename.py            # one ADR in flight
    python3 scripts/adr_rename.py <slug>     # pick one of several

Renames ``docs/adr/XXXX-<slug>.md`` to ``docs/adr/<pr>-<slug>.md``, fixes the
heading, repoints citations at the new number, and re-runs the guard.

Naming a numbered ADR renumbers it, which is how a collision gets resolved:
two branches that each claimed 0017 cannot both keep it, and the one that has
not landed takes its pull request number instead.

Three citation forms move together: ``ADR-<slug>``, the old ``ADR-<number>``
(or ``ADR-XXXX``), and any relative link to the old filename. The middle form
is the one worth being thorough about. A slug citation still resolves if the
sweep misses it, and a dead relative link is reported by the guard -- but a
stale number resolves *silently to whichever ADR now holds it*, and citations
in code are outside the guard's reach entirely, so nothing would report them.

Which is also why an old number is swept only when no other ADR still holds
it. If one does, the citations are genuinely ambiguous -- some may already
mean the other ADR -- so they are listed for review rather than rewritten.

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

# Mirrors FILENAME_RE in adr_check.py: a leading number or the placeholder,
# then the slug. Used here to read a filename back apart.
SOURCE_RE = re.compile(
    rf"^(?P<previous>\d{{4,}}|{PLACEHOLDER})-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.md$"
)


def unnumbered() -> list[Path]:
    return sorted(ADR_DIR.glob(f"{PLACEHOLDER}-*.md"))


def numbered_with_slug(slug: str) -> list[Path]:
    """Every already-numbered ADR carrying exactly this slug.

    The glob alone is too loose -- `*-two.md` also matches `0012-not-two.md` --
    so each candidate is parsed and its slug compared exactly.
    """
    found = []
    for path in sorted(ADR_DIR.glob(f"*-{slug}.md")):
        match = SOURCE_RE.match(path.name)
        if match and match["slug"] == slug and match["previous"] != PLACEHOLDER:
            found.append(path)
    return found


def claimant(number: str, exclude: Path) -> Path | None:
    """Another ADR still holding `number`, if one exists.

    Its presence makes citations of that number ambiguous: some may mean this
    ADR and some the other, and no rewrite can tell them apart.
    """
    if number == PLACEHOLDER:
        return None
    for path in sorted(ADR_DIR.rglob("*.md")):
        if path == exclude:
            continue
        match = SOURCE_RE.match(path.name)
        if match and match["previous"] != PLACEHOLDER:
            if int(match["previous"]) == int(number):
                return path
    return None


def pick_source(slug: str | None) -> Path | None:
    """The ADR to number, or None after reporting why it is ambiguous."""
    if slug is not None:
        src = ADR_DIR / f"{PLACEHOLDER}-{slug}.md"
        if src.exists():
            return src
        # Naming a numbered ADR explicitly is the renumber path. It is never
        # reached without an argument, so no bare run can renumber a landed
        # ADR by accident.
        renumber = numbered_with_slug(slug)
        if len(renumber) == 1:
            return renumber[0]
        if len(renumber) > 1:
            sys.stderr.write(f"more than one ADR has the slug {slug!r}:\n")
            for path in renumber:
                sys.stderr.write(f"  {path}\n")
            return None
        sys.stderr.write(
            f"{src} does not exist, and no numbered ADR has the slug {slug!r}\n"
        )
        return None

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


def previous_citation_re(previous: str) -> re.Pattern[str]:
    """Match citations of the number an ADR is moving away from.

    `ADR-0*17` also catches the `ADR-17` short form the guard reports as a
    typo, and the `(?!-\\d)` guard mirrors adr_check.py so a hyphenated date
    like `ADR-2026-08-10` is never read as a citation of ADR-2026.
    """
    if previous == PLACEHOLDER:
        return re.compile(rf"ADR-{PLACEHOLDER}\b")
    return re.compile(rf"ADR-0*{int(previous)}\b(?!-\d)")


def previous_citation_search(previous: str) -> str:
    """The same thing, reduced to what `git grep -E` can parse.

    POSIX ERE has no lookahead, so the precise pattern above cannot be handed
    to git. This is deliberately looser: it only has to find candidate files,
    and `previous_citation_re` decides what actually gets rewritten. Passing
    the lookahead form would make git grep fail and the sweep silently find
    nothing at all.
    """
    if previous == PLACEHOLDER:
        return rf"ADR-{PLACEHOLDER}"
    return rf"ADR-0*{int(previous)}"


def sweep_references(
    slug: str,
    number: str,
    previous: str = PLACEHOLDER,
    *,
    include_previous: bool = True,
) -> int:
    """Point every citation of an ADR at its new number.

    Three forms move together: the prose citation ``ADR-<slug>``, the old
    ``ADR-<number>``, and any relative link to the old filename. The link is
    the one that must be exact -- the file has moved, so a missed link is a
    dead link -- but the old number is the one that fails quietly, including
    in code the guard never reads.

    `include_previous` is False when another ADR still holds `previous`, which
    makes those citations ambiguous rather than merely stale.
    """
    # The lookahead stops a slug that prefixes a longer one from matching it:
    # sweeping `tee-node` must leave `ADR-tee-node-deferred-writes` alone.
    rewrites = [
        (re.compile(rf"ADR-{re.escape(slug)}(?![a-z0-9-])"), f"ADR-{number}"),
        (
            re.compile(
                rf"(?<=\()\s*{re.escape(previous)}-{re.escape(slug)}\.md(?=[)#])"
            ),
            f"{number}-{slug}.md",
        ),
    ]
    searches = [
        rf"ADR-{re.escape(slug)}",
        rf"{re.escape(previous)}-{re.escape(slug)}\.md",
    ]
    if include_previous:
        rewrites.append((previous_citation_re(previous), f"ADR-{number}"))
        searches.append(previous_citation_search(previous))

    found = subprocess.run(
        ["git", "grep", "--untracked", "-lIE", "|".join(searches)],
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

    match = SOURCE_RE.match(src.name)
    if match is None:
        sys.stderr.write(f"{src} is not a well-formed ADR filename\n")
        return 1
    previous, slug = match["previous"], match["slug"]

    if previous == number:
        sys.stdout.write(f"{src} already carries number {number}\n")
        return 0

    dest = ADR_DIR / f"{number}-{slug}.md"

    # Decided before the move, while `src` is still the file to exclude.
    other = claimant(previous, src)

    subprocess.run(["git", "mv", str(src), str(dest)], check=True)
    dest.write_text(
        dest.read_text(encoding="utf-8").replace(
            f"# ADR-{previous}:", f"# ADR-{number}:", 1
        ),
        encoding="utf-8",
    )
    sys.stdout.write(f"renamed to {dest}\n")

    changed = sweep_references(slug, number, previous, include_previous=other is None)
    sys.stdout.write(
        f"rewrote citations to ADR-{number} in {changed} file(s)\n"
        if changed
        else f"no references to ADR-{slug} to rewrite\n"
    )

    if other is not None:
        sys.stdout.write(
            f"\nleft ADR-{previous} citations alone: {other.name} still holds that\n"
            "number, so a citation of it may mean either ADR. Review them by hand:\n"
        )
        listing = subprocess.run(
            ["git", "grep", "--untracked", "-nIE", previous_citation_search(previous)],
            capture_output=True,
            text=True,
        )
        for line in listing.stdout.splitlines():
            sys.stdout.write(f"  {line}\n")

    # The directory checks read the working tree, so they see what just
    # happened: the new filename, the heading, the citations, the links.
    #
    # The added-ADR checks are deliberately not run here. They diff
    # `<base>...HEAD`, which is committed state, and the rename is not
    # committed yet -- so they would judge the *old* filename. In the
    # placeholder flow that is merely inert (a placeholder has no number to
    # check); when renumbering it is actively wrong, reporting the number the
    # rename just removed. CI runs them where they mean something.
    #
    # Located next to this file rather than by relative path, so the guard is
    # found no matter where the caller invoked us from.
    code = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().with_name("adr_check.py"))],
    ).returncode
    sys.stdout.write(
        f"\ncommit the rename, then the full guard applies:\n"
        f"  python3 scripts/adr_check.py --base {base} --pr {number}\n"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
