"""Validate ADR filenames, numbering, and cross-references.

Enforces the numbering scheme documented in ``docs/adr/README.md``: ADRs
numbered below ``PR_NUMBER_FLOOR`` are frozen legacy sequential numbers,
everything at or above it is the number of the pull request that adds the ADR.
Deriving the number from the forge means no two branches can claim the same one.

Stdlib only, so CI runs it without installing the project.

    python3 scripts/adr_check.py                    # whole-directory checks
    python3 scripts/adr_check.py --base main        # add the new-ADR checks
    python3 scripts/adr_check.py --base main --pr 2211
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ADR_DIR = Path("docs/adr")

# Meta documents: they carry illustrative numbers that intentionally do not
# resolve to real ADRs, so they are exempt from the reference checks.
META_FILES = frozenset({"template.md", "README.md"})

PLACEHOLDER = "XXXX"

# Numbers below the floor are legacy sequential ADRs (0002-0017 at the time
# this landed). They stay at their original values forever: roughly sixty code
# comments across python/xorq cite them by number, and renumbering would either
# break those citations or require a risky sweep of them.
PR_NUMBER_FLOOR = 1000

# Legacy-numbered ADRs still in flight on branches that were opened before the
# PR-number scheme landed. Each entry suppresses the "new ADRs must use a pull
# request number" error for exactly one number. Delete an entry once its branch
# merges. This list must only ever shrink.
LEGACY_IN_FLIGHT = frozenset(
    {
        1,  # 0001-git-annex-over-git-lfs, on origin/perf/catalog/use-git-annex
        18,  # 0018-content-store-capability-and-binding, on hosted-presigned-catalogs-fixes
        20,  # added by PR #2200
        21,  # added by PR #2200
        23,  # added by PR #2200
        19,  # reserved by a later entry of the #2200 stack
        22,  # reserved by a later entry of the #2200 stack
        24,  # reserved by a later entry of the #2200 stack
        25,  # reserved by a later entry of the #2200 stack
    }
)

FILENAME_RE = re.compile(
    rf"^(?P<num>\d{{4,}}|{PLACEHOLDER})-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.md$"
)
HEADING_RE = re.compile(rf"^#\s+ADR-(?P<num>\d{{4,}}|{PLACEHOLDER})\s*:", re.MULTILINE)
# An ADR is cited either by number (ADR-0011) or by slug
# (ADR-catalog-single-git-remote). The two are lexically disjoint -- a number
# starts with a digit, a slug with a letter -- so one pattern reads both.
#
# The lookahead applies to the numeric branch only: it keeps a hyphenated date
# such as "ADR-2026-08-08" from reading as a reference to ADR-2026 (prose about
# date-based numbering hits this). Slugs are letter-initial and cannot match a
# date, so they need no such guard.
REFERENCE_RE = re.compile(r"ADR-(?:(?P<num>\d{4,})(?!-\d)|(?P<slug>[a-z][a-z0-9-]*))")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)#\s]+\.md)[^)]*\)")


class Problems:
    """Collects annotated failures so one run reports every problem."""

    def __init__(self) -> None:
        self.count = 0
        self.warnings = 0

    def add(self, path: Path | str, message: str) -> None:
        self.count += 1
        # GitHub renders this as an inline annotation on the file; plain text
        # everywhere else. Mirrors `ruff --output-format=github` in ci-lint.
        sys.stdout.write(f"::error file={path}::{message}\n")
        sys.stderr.write(f"{path}: {message}\n")

    def warn(self, path: Path | str, message: str) -> None:
        """Report without failing the run.

        Used for named references that do not resolve, which are legitimate
        while the ADR they name is still on an unlanded branch.
        """
        self.warnings += 1
        sys.stdout.write(f"::warning file={path}::{message}\n")
        sys.stderr.write(f"{path}: warning: {message}\n")


def adr_files() -> list[Path]:
    return sorted(p for p in ADR_DIR.glob("*.md") if p.name not in META_FILES)


def parse_name(name: str) -> tuple[int | None, str]:
    """Split an ADR filename into its number and its slug.

    The number is None while the file still carries the placeholder. The slug
    is the ADR's identity: it is fixed when the file is created and never
    changes, which is what lets an ADR be cited before it has a number.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(name)
    number = match.group("num")
    return (None if number == PLACEHOLDER else int(number), match.group("slug"))


def parse_number(name: str) -> int | None:
    """Return the numeric prefix of an ADR filename, or None if it is XXXX."""
    return parse_name(name)[0]


def check_directory(problems: Problems) -> dict[int, Path]:
    """Validate every ADR on disk. Returns the number to path index."""
    by_number: dict[int, Path] = {}
    by_slug: dict[str, Path] = {}
    checkable: list[tuple[Path, int | None]] = []

    for path in adr_files():
        try:
            number, slug = parse_name(path.name)
        except ValueError:
            problems.add(
                path,
                "filename does not match NNNN-slug.md (four or more digits, or "
                f"{PLACEHOLDER} before the pull request number is known), where "
                "slug is lowercase words joined by single hyphens",
            )
            continue

        if number is None:
            # Reported, but the file is still indexed and still checked below:
            # an unnumbered ADR is citable by slug, and its own references are
            # as worth validating as any other's.
            problems.add(
                path,
                f"still carries the {PLACEHOLDER} placeholder. Open the pull "
                "request, then run `python3 scripts/adr_rename.py` to give it "
                "the pull "
                "request number",
            )
        elif (previous := by_number.get(number)) is not None:
            problems.add(
                path,
                f"number {number:04d} is already used by {previous.name}. Two "
                "branches claimed the same number; rename the newer ADR",
            )
            continue
        else:
            by_number[number] = path

        if (previous := by_slug.get(slug)) is not None:
            problems.add(
                path,
                f"slug `{slug}` is already used by {previous.name}. The slug is "
                "an ADR's identity and what named references resolve against, "
                "so it has to be unique; rename the newer ADR",
            )
            continue
        by_slug[slug] = path
        checkable.append((path, number))

    # Second pass: both indexes must be complete before references resolve.
    for path, number in checkable:
        text = path.read_text(encoding="utf-8")
        check_heading(problems, path, number, text)
        check_references(problems, path, text, by_number, by_slug)

    return by_number


def check_heading(
    problems: Problems, path: Path, number: int | None, text: str
) -> None:
    expected = PLACEHOLDER if number is None else f"{number:04d}"
    match = HEADING_RE.search(text)
    if match is None:
        problems.add(path, "missing an `# ADR-NNNN: <title>` heading")
    elif match.group("num") != expected:
        problems.add(
            path,
            f"heading says ADR-{match.group('num')} but the filename says "
            f"{expected}; they must agree",
        )


def check_references(
    problems: Problems,
    path: Path,
    text: str,
    by_number: dict[int, Path],
    by_slug: dict[str, Path],
) -> None:
    """Resolve every ADR reference and every relative .md link.

    A numbered reference that does not resolve is an error: the number is
    allocated by the forge, so a missing one is a mistake. A named reference
    that does not resolve is only a warning, because the ADR it names may
    legitimately still be on an unlanded branch -- this is what makes forward
    references possible without reserving numbers ahead of time.
    """
    numbered: set[str] = set()
    named: set[str] = set()
    for match in REFERENCE_RE.finditer(text):
        if (number := match.group("num")) is not None:
            numbered.add(number)
        else:
            named.add(match.group("slug"))

    for reference in sorted(numbered):
        if int(reference) not in by_number:
            problems.add(path, f"references ADR-{reference}, which does not exist")

    for reference in sorted(named):
        if reference not in by_slug:
            problems.warn(
                path,
                f"references ADR-{reference}, which is not in this directory. "
                "That is expected if it lands in a later pull request; "
                "otherwise the slug is wrong",
            )

    for target in sorted(set(MD_LINK_RE.findall(text))):
        if "/" in target or target.startswith("."):
            continue  # only same-directory ADR links are in scope here
        if not (ADR_DIR / target).exists():
            problems.add(path, f"links to {target}, which does not exist")


def added_adrs(base: str) -> list[Path]:
    """ADR files this branch adds relative to the merge base with `base`."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=A",
            f"{base}...HEAD",
            "--",
            str(ADR_DIR),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        Path(line)
        for line in result.stdout.splitlines()
        if line and Path(line).name not in META_FILES
    ]


def _safe_number(path: Path) -> int | None:
    """parse_number that reports nothing; malformed names fall through to the
    directory checks, which already name the problem precisely."""
    try:
        return parse_number(path.name)
    except ValueError:
        return None


def check_added(problems: Problems, added: list[Path], pr: int | None) -> None:
    # A batch of allowlisted legacy ADRs is exempt from one-per-pull-request:
    # those branches predate the scheme and their numbers were already claimed.
    # Anything involving a new-scheme ADR is still held to one per pull request.
    legacy_batch = all(_safe_number(p) in LEGACY_IN_FLIGHT for p in added)
    if len(added) > 1 and not legacy_batch:
        names = ", ".join(sorted(p.name for p in added))
        problems.add(
            ADR_DIR,
            f"this pull request adds {len(added)} ADRs ({names}), but the number "
            "comes from the pull request, so only one ADR can be added per pull "
            "request. Move the others to their own pull requests",
        )
        return

    for path in added:
        try:
            number = parse_number(path.name)
        except ValueError:
            continue  # already reported by the directory checks
        if number is None:
            continue  # placeholder already reported

        if number < PR_NUMBER_FLOOR:
            if number in LEGACY_IN_FLIGHT:
                continue
            problems.add(
                path,
                "new ADRs take the number of the pull request that adds them, "
                f"which is at least {PR_NUMBER_FLOOR}. Numbers below that are "
                "frozen legacy ADRs. Run `python3 scripts/adr_rename.py`",
            )
        elif pr is not None and number != pr:
            problems.add(
                path,
                f"numbered {number} but added by pull request {pr}. The ADR "
                "number must equal the pull request number: run "
                "`python3 scripts/adr_rename.py`",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        help="base ref or sha to diff against; enables the new-ADR checks",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="pull request number the new ADR must be named after",
    )
    args = parser.parse_args()

    if not ADR_DIR.is_dir():
        sys.stderr.write(f"{ADR_DIR} not found; run from the repository root\n")
        return 2

    problems = Problems()
    check_directory(problems)
    if args.base:
        check_added(problems, added_adrs(args.base), args.pr)

    if problems.warnings:
        sys.stderr.write(
            f"\n{problems.warnings} unresolved named reference(s). Not a "
            "failure: an ADR named by slug may still be on an unlanded branch.\n"
        )
    if problems.count:
        sys.stderr.write(
            f"\n{problems.count} ADR problem(s). See docs/adr/README.md for the "
            "numbering rules.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
