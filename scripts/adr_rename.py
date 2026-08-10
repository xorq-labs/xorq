"""Give the in-flight ADR the number of this branch's pull request.

    python3 scripts/adr_rename.py            # one ADR in flight
    python3 scripts/adr_rename.py <slug>     # pick one of several

Renames ``docs/adr/XXXX-<slug>.md`` to ``docs/adr/<pr>-<slug>.md``, fixes the
heading, repoints citations at the new number, and re-runs the guard.

Naming a numbered ADR renumbers it, which is how a collision gets resolved:
two branches that each claimed 0017 cannot both keep it, and the one that has
not landed takes its pull request number instead.

Three citation forms move together: ``ADR-<slug>``, the old ``ADR-<number>``,
and any relative link to the old filename. The middle form is the one worth
being thorough about. A slug citation still resolves if the sweep misses it,
and a dead relative link is reported by the guard -- but a stale number
resolves *silently to whichever ADR now holds it*, and citations in code are
outside the guard's reach entirely, so nothing would report them.

Which is also why an old number is swept only when no other ADR still holds
it. If one does, the citations are genuinely ambiguous -- some may already
mean the other ADR -- so they are listed for review rather than rewritten.

``ADR-XXXX`` is deliberately not a fourth form. The placeholder is not an
identifier: template.md carries it permanently, and so does every other draft
in flight, so a repository-wide rewrite of it would renumber all of them. It
is rewritten only inside the file being numbered.

Stdlib only, matching scripts/adr_check.py.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# The guard owns the filename grammar; read it back apart with the same regex
# rather than a copy, so the two cannot drift into disagreeing about what a
# valid ADR filename is. A plain sibling import: python puts this script's
# directory on sys.path, and pytest reaches it the same way (see
# scripts/tests/test_adr_check.py).
from adr_check import (
    ADR_DIR,
    CODE_FENCE_RE,
    FILENAME_RE,
    INLINE_CODE_RE,
    PLACEHOLDER,
    URL_RE,
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
        match = FILENAME_RE.match(path.name)
        if match and match["slug"] == slug and match["num"] != PLACEHOLDER:
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
        match = FILENAME_RE.match(path.name)
        if match and match["num"] != PLACEHOLDER:
            if int(match["num"]) == int(number):
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
    try:
        result = subprocess.run(
            ["gh", "pr", "view", "--json", "number,baseRefName"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # The number has to come from the forge, and gh is how this script
        # reaches it. Say what to do instead rather than raising.
        sys.stderr.write(
            "the GitHub CLI (gh) is not installed, so the pull request number "
            "cannot be read. Install it, or rename the file by hand to "
            "<pr>-<slug>.md and make the heading agree\n"
        )
        return None
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
    typo. The leading `\\b` supplies the boundary the number cannot: without it
    `BADR-0017` reads as a citation. The trailing `(?!-\\d)` mirrors
    adr_check.py, so a hyphenated date like `ADR-2026-08-10` is never read as a
    citation of ADR-2026.

    Never called with the placeholder, which `int()` would reject anyway --
    see the note on that in `sweep_references`.
    """
    return re.compile(rf"\bADR-0*{int(previous)}\b(?!-\d)")


def previous_citation_search(previous: str) -> str:
    """The same thing, reduced to what `git grep -E` can parse.

    POSIX ERE has no lookahead, so the precise pattern above cannot be handed
    to git, and `\\b` is a GNU extension this would rather not depend on. This
    is deliberately looser: it only has to find candidate files, and
    `previous_citation_re` decides what actually gets rewritten. Passing the
    lookahead form would make git grep fail and the sweep silently find nothing
    at all.
    """
    return rf"ADR-0*{int(previous)}"


def shown_spans(text: str) -> list[tuple[int, int]]:
    """Merged spans of the regions adr_check.py's `strip_shown_code` drops.

    A fence, an inline-code span, and a URL display a citation rather than make
    one, which is why the guard does not resolve references inside them. The
    sweep honours the same boundary: docs/adr/README.md and CONTRIBUTING.md
    both teach the named form by showing one, and numbering the ADR an example
    happens to name should not edit the prose that explains the convention.

    (Which is why no example slug is written out here: this file is Python, so
    the sweep reads it as code and would rewrite one.)
    """
    spans = [
        match.span()
        for pattern in (CODE_FENCE_RE, INLINE_CODE_RE, URL_RE)
        for match in pattern.finditer(text)
    ]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        # A URL inside a fence produces a span inside a span; keep the outer.
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def apply_rewrites(
    text: str, rewrites: list[tuple[re.Pattern[str], str]], *, protect_shown: bool
) -> str:
    """Apply every rewrite, optionally leaving shown-code regions untouched.

    `protect_shown` is on for Markdown and off for everything else. A backtick
    in a Python docstring is not the same claim as one in prose, and a stale
    number in code is precisely the failure this sweep exists to prevent -- so
    code is rewritten throughout.
    """

    def rewrite(chunk: str) -> str:
        for pattern, replacement in rewrites:
            chunk = pattern.sub(replacement, chunk)
        return chunk

    if not protect_shown:
        return rewrite(text)

    out: list[str] = []
    cursor = 0
    for start, end in shown_spans(text):
        out.append(rewrite(text[cursor:start]))
        out.append(text[start:end])
        cursor = end
    out.append(rewrite(text[cursor:]))
    return "".join(out)


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

    What makes a repository-wide rewrite safe is that every form swept here
    identifies one ADR: two name the slug outright, and the third is a number
    no other ADR holds, which `include_previous` is how the caller confirms.
    `ADR-XXXX` has neither property -- template.md carries it permanently, and
    so does every other draft in flight -- so the placeholder is never swept
    from here. `main` rewrites it inside the file being numbered, which is the
    one place it does mean a particular ADR.
    """
    # The lookahead stops a slug that prefixes a longer one from matching it:
    # sweeping `tee-node` must leave `ADR-tee-node-deferred-writes` alone. The
    # leading `\b` does the same on the other side.
    rewrites = [
        (re.compile(rf"\bADR-{re.escape(slug)}(?![a-z0-9-])"), f"ADR-{number}"),
        # The link is matched with its opening paren so a bare filename in
        # prose is left alone, and with any leading space put back, so a title
        # -- `(0011-x.md "Why")` -- survives the rewrite intact.
        (
            re.compile(
                rf"(?<=\()(?P<lead>\s*){re.escape(previous)}-{re.escape(slug)}"
                r"\.md(?=[)#\s])"
            ),
            rf"\g<lead>{number}-{slug}.md",
        ),
    ]
    searches = [
        rf"ADR-{re.escape(slug)}",
        rf"{re.escape(previous)}-{re.escape(slug)}\.md",
    ]
    if include_previous and previous != PLACEHOLDER:
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
        rewritten = apply_rewrites(text, rewrites, protect_shown=path.suffix == ".md")
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed += 1
    return changed


def move(src: Path, dest: Path) -> bool:
    """Move the ADR, reporting rather than raising when git objects.

    `git mv` exits 128 on a file it does not track, which is the state
    `adr_new.py` leaves behind: running the rename before committing the draft
    is an easy first mistake, and a traceback is a poor way to describe it.
    There is nothing for git to do with an untracked file, so it is simply
    moved.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(src)],
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        src.rename(dest)
        return True

    result = subprocess.run(
        ["git", "mv", str(src), str(dest)], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or f"could not move {src} to {dest}\n")
        return False
    return True


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

    match = FILENAME_RE.match(src.name)
    if match is None:
        sys.stderr.write(f"{src} is not a well-formed ADR filename\n")
        return 1
    # The number in the source filename is the one being moved away from.
    previous, slug = match["num"], match["slug"]

    if previous == number:
        sys.stdout.write(f"{src} already carries number {number}\n")
        return 0

    dest = ADR_DIR / f"{number}-{slug}.md"
    if dest.exists():
        sys.stderr.write(f"{dest} already exists, so {src.name} cannot take it\n")
        return 1

    # Decided before the move, while `src` is still the file to exclude.
    other = claimant(previous, src)

    if not move(src, dest):
        return 1

    moved = dest.read_text(encoding="utf-8")
    if previous == PLACEHOLDER:
        # Every ADR-XXXX in this file, not only the heading: a draft may cite
        # itself while it waits for a number. Confined to this file because the
        # placeholder identifies nothing -- see the note in `sweep_references`.
        moved = apply_rewrites(
            moved,
            [(re.compile(rf"\bADR-{PLACEHOLDER}\b"), f"ADR-{number}")],
            protect_shown=True,
        )
    else:
        # Only the heading. Other citations of the old number are the sweep's
        # business, which knows to leave them alone if they are ambiguous.
        moved = moved.replace(f"# ADR-{previous}:", f"# ADR-{number}:", 1)
    dest.write_text(moved, encoding="utf-8")
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
