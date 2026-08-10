"""Validate ADR filenames, numbering, and cross-references.

Enforces the numbering scheme documented in ``docs/adr/README.md``: ADRs
numbered below ``PR_NUMBER_FLOOR`` are frozen legacy sequential numbers,
everything at or above it is the number of the pull request that adds the ADR.
Deriving the number from the forge means no two branches can claim the same one.

Stdlib only, so CI runs it without installing the project.

    python3 scripts/adr_check.py                    # whole-directory checks
    python3 scripts/adr_check.py --base main        # add the new-ADR checks
    python3 scripts/adr_check.py --base main --pr 2211
    python3 scripts/adr_check.py --format github    # inline CI annotations
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
# this landed). They stay at their original values forever: about fifty code
# comments across python/xorq cite them by number, and renumbering would either
# break those citations or require a risky sweep of them.
PR_NUMBER_FLOOR = 1000

# Legacy-numbered ADRs still in flight on branches that were opened before the
# PR-number scheme landed, as {number: slug}. Each entry exempts exactly one
# file -- the number alone is not enough, or the entry becomes a general licence
# to add any ADR at that number. Delete an entry once its branch merges. This
# mapping must only ever shrink.
#
# Numbers are NOT held here for ADRs that do not exist yet. Reserving a number
# ahead of the file is the coordination this scheme exists to remove; an ADR
# still being written takes the number of the pull request that adds it, and is
# cited by slug until then.
LEGACY_IN_FLIGHT = {
    1: "git-annex-over-git-lfs",  # origin/perf/catalog/use-git-annex
    18: "content-store-capability-and-binding",  # hosted-presigned-catalogs-fixes
    20: "engine-behavior-as-immutable-identity-folded-spec",  # PR #2200
    21: "engine-construction-is-two-level-identityspec-feeds-enginebuilder",  # PR #2200
    23: "identity-spec-contributions-are-entry-points-composed-order-independently",  # PR #2200
}

FILENAME_RE = re.compile(
    rf"^(?P<num>\d{{4,}}|{PLACEHOLDER})-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.md$"
)
HEADING_RE = re.compile(rf"^#\s+ADR-(?P<num>\d{{4,}}|{PLACEHOLDER})\s*:", re.MULTILINE)

# An ADR is cited either by number (ADR-0011) or by slug
# (ADR-catalog-single-git-remote). The two are lexically disjoint -- a number
# starts with a digit, a slug with a letter -- so one pattern reads both.
#
# The `short` branch catches ADR-11 and ADR-999: not valid citations, since
# numbers are written with at least four digits, but the obvious typo. Ignoring
# them silently would leave a broken reference unreported.
#
# The negative lookahead on `num` keeps a hyphenated date such as
# "ADR-2026-08-08" from reading as a reference to ADR-2026 (prose about
# date-based numbering hits this). Slugs are letter-initial and cannot match a
# date, so they need no such guard.
REFERENCE_RE = re.compile(
    r"ADR-(?:(?P<num>\d{4,})(?!-\d)|(?P<short>\d{1,3})(?!\d)|(?P<slug>[a-z][a-z0-9-]*))"
)
MD_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)#\s]+\.md)[^)]*\)")

# Regions that display an ADR reference rather than make one. A document about
# the numbering scheme necessarily prints numbers that do not resolve, and CI
# output pasted into an ADR would otherwise fail the very check that produced
# it. Stripped before references and links are collected.
CODE_FENCE_RE = re.compile(
    r"^(?P<fence>```|~~~).*?^(?P=fence)", re.MULTILINE | re.DOTALL
)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
URL_RE = re.compile(r"<?https?://[^\s>)]+>?")


class Problems:
    """Collects annotated failures so one run reports every problem."""

    def __init__(self, github: bool = False) -> None:
        self.count = 0
        self.warnings = 0
        # GitHub renders `::error file=...` as an inline annotation on the file.
        # Off by default so local runs are readable; ci-adr.yml opts in, the way
        # ci-lint passes `ruff --output-format=github`.
        self.github = github

    def add(self, path: Path | str, message: str) -> None:
        self.count += 1
        self._emit("error", path, message)

    def warn(self, path: Path | str, message: str) -> None:
        """Report without failing the run.

        Used for named references that do not resolve, which are legitimate
        while the ADR they name is still on an unlanded branch.
        """
        self.warnings += 1
        self._emit("warning", path, message)

    def _emit(self, level: str, path: Path | str, message: str) -> None:
        if self.github:
            sys.stdout.write(f"::{level} file={path}::{message}\n")
        prefix = "" if level == "error" else f"{level}: "
        sys.stderr.write(f"{path}: {prefix}{message}\n")


def adr_files() -> list[Path]:
    """Every ADR in the directory, including any nested by mistake.

    The glob is recursive so a file in a subdirectory is checked rather than
    skipped. `added_adrs()` diffs the whole tree, so a nested file that the
    directory pass never saw would be a hole in the guard, not an absence.
    """
    return sorted(p for p in ADR_DIR.rglob("*.md") if p.name not in META_FILES)


def parse_name(name: str) -> tuple[int | None, str, str]:
    """Split an ADR filename into its number, slug, and literal number text.

    The number is None while the file still carries the placeholder. The slug
    is the ADR's identity: it is fixed when the file is created and never
    changes, which is what lets an ADR be cited before it has a number. The
    literal text is kept so messages can quote the filename as written.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(name)
    raw = match.group("num")
    number = None if raw == PLACEHOLDER else int(raw)
    return (number, match.group("slug"), raw)


def parse_number(name: str) -> int | None:
    """Return the numeric prefix of an ADR filename, or None if it is XXXX."""
    return parse_name(name)[0]


def strip_shown_code(text: str) -> str:
    """Drop regions that display an ADR reference rather than make one."""
    for pattern in (CODE_FENCE_RE, INLINE_CODE_RE, URL_RE):
        text = pattern.sub(" ", text)
    return text


def check_directory(problems: Problems) -> None:
    """Validate every ADR on disk."""
    by_number: dict[int, Path] = {}
    by_slug: dict[str, Path] = {}
    checkable: list[tuple[Path, int | None, str]] = []

    for path in adr_files():
        if path.parent != ADR_DIR:
            problems.add(
                path,
                f"ADRs live directly in {ADR_DIR}/, not in subdirectories. The "
                "numbering and reference checks treat the directory as flat",
            )
            continue

        try:
            number, slug, raw = parse_name(path.name)
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
                "the pull request number",
            )
        elif (previous := by_number.get(number)) is not None:
            problems.add(
                path,
                f"number {number:04d} is already used by {previous.name}. Two "
                "branches claimed the same number; rename the newer ADR",
            )
        else:
            by_number[number] = path

        if (previous := by_slug.get(slug)) is not None:
            problems.add(
                path,
                f"slug `{slug}` is already used by {previous.name}. The slug is "
                "an ADR's identity and what named references resolve against, "
                "so it has to be unique; rename the newer ADR",
            )
        else:
            by_slug[slug] = path

        # Appended even when a duplicate was reported above: a file with a
        # clashing number can still have a broken heading or a dead link, and
        # reporting one problem per run makes for a slow fix loop.
        checkable.append((path, number, raw))

    # Second pass: both indexes must be complete before references resolve.
    for path, number, raw in checkable:
        text = path.read_text(encoding="utf-8")
        check_heading(problems, path, number, raw, text)
        check_references(problems, path, text, by_number, by_slug)


def check_heading(
    problems: Problems, path: Path, number: int | None, raw: str, text: str
) -> None:
    match = HEADING_RE.search(text)
    if match is None:
        problems.add(path, "missing an `# ADR-NNNN: <title>` heading")
    elif match.group("num") != raw:
        problems.add(
            path,
            f"heading says ADR-{match.group('num')} but the filename says "
            f"{raw}; they must agree",
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
    prose = strip_shown_code(text)

    numbered: set[str] = set()
    short: set[str] = set()
    named: set[str] = set()
    for match in REFERENCE_RE.finditer(prose):
        if (number := match.group("num")) is not None:
            numbered.add(number)
        elif (stub := match.group("short")) is not None:
            short.add(stub)
        else:
            named.add(match.group("slug"))

    for reference in sorted(numbered):
        if int(reference) not in by_number:
            problems.add(path, f"references ADR-{reference}, which does not exist")

    for reference in sorted(short, key=int):
        problems.add(
            path,
            f"references ADR-{reference}, but ADR numbers are written with at "
            f"least four digits. Did you mean ADR-{int(reference):04d}?",
        )

    for reference in sorted(named):
        if reference not in by_slug:
            problems.warn(
                path,
                f"references ADR-{reference}, which is not in this directory. "
                "That is expected if it lands in a later pull request; "
                "otherwise the slug is wrong",
            )

    for target in sorted(set(MD_LINK_RE.findall(prose))):
        if "/" in target or target.startswith("."):
            continue  # only same-directory ADR links are in scope here
        if not (ADR_DIR / target).exists():
            problems.add(path, f"links to {target}, which does not exist")


def added_adrs(base: str) -> list[Path] | None:
    """ADR files this branch adds relative to the merge base with `base`.

    None means the base ref could not be resolved -- an unfetched sha, a
    shallow clone, or a branch this clone does not have. The message is written
    here; the caller turns it into an exit code.
    """
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
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        sys.stderr.write(
            f"cannot diff against {base!r}: "
            f"{detail[-1] if detail else 'git diff failed'}. "
            "Pass a ref this clone has, and check that the checkout is not "
            "shallow (ci-adr.yml sets fetch-depth: 0 for this reason)\n"
        )
        return None
    return [
        Path(line)
        for line in result.stdout.splitlines()
        if line and Path(line).name not in META_FILES
    ]


def _safe_name(path: Path) -> tuple[int | None, str, str] | None:
    """parse_name that reports nothing; a malformed name falls through to the
    directory checks, which already name the problem precisely."""
    try:
        return parse_name(path.name)
    except ValueError:
        return None


def is_allowlisted(path: Path) -> bool:
    """True for a legacy-numbered ADR this scheme deliberately lets through.

    Both the number and the slug have to match. Keying on the number alone
    would turn each entry into a licence to add *any* ADR at that number.
    """
    parsed = _safe_name(path)
    if parsed is None:
        return False
    number, slug, _ = parsed
    return number is not None and LEGACY_IN_FLIGHT.get(number) == slug


def check_added(problems: Problems, added: list[Path], pr: int | None) -> None:
    # A batch of allowlisted legacy ADRs is exempt from one-per-pull-request:
    # those branches predate the scheme and their files were already written
    # against those numbers. Anything else is held to one ADR per pull request.
    if len(added) > 1 and not all(is_allowlisted(p) for p in added):
        names = ", ".join(sorted(p.name for p in added))
        problems.add(
            ADR_DIR,
            f"this pull request adds {len(added)} ADRs ({names}), but the number "
            "comes from the pull request, so only one ADR can be added per pull "
            "request. Move the others to their own pull requests -- splitting "
            "costs nothing, because an ADR is citable by slug from the moment it "
            "is written: cite `ADR-<slug>` rather than waiting for a number or "
            "reserving one",
        )
        return

    for path in added:
        parsed = _safe_name(path)
        if parsed is None:
            continue  # already reported by the directory checks
        number, slug, _ = parsed
        if number is None:
            continue  # placeholder already reported

        if number < PR_NUMBER_FLOOR:
            held = LEGACY_IN_FLIGHT.get(number)
            if held == slug:
                continue
            if held is not None:
                problems.add(
                    path,
                    f"number {number:04d} belongs to `{held}`, which is still in "
                    "flight on another branch. New ADRs take the number of the "
                    "pull request that adds them: run "
                    "`python3 scripts/adr_rename.py`",
                )
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
    parser.add_argument(
        "--format",
        choices=("text", "github"),
        default="text",
        help="github adds ::error/::warning inline annotations for CI",
    )
    args = parser.parse_args()

    if not ADR_DIR.is_dir():
        sys.stderr.write(f"{ADR_DIR} not found; run from the repository root\n")
        return 2

    problems = Problems(github=args.format == "github")
    check_directory(problems)
    if args.base:
        added = added_adrs(args.base)
        if added is None:
            return 2
        check_added(problems, added, args.pr)

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
