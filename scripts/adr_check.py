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

# Documents about ADRs rather than ADRs. They carry no number, so they cannot
# satisfy the filename grammar, and the numbers they show are illustrative.
# Skipped wholesale rather than reported as malformed.
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
# to add any ADR at that number.
#
# This mapping must only ever shrink, and `check_allowlist` is what makes that
# more than a wish: once an entry's ADR is in the tree, the entry has done its
# job and is reported until someone deletes it. The deletion is always a
# follow-up, never part of the pull request that lands the ADR -- CI reads the
# merged code, so an entry removed there would reject the file it was
# protecting.
#
# Numbers are NOT held here for ADRs that do not exist yet. Reserving a number
# ahead of the file is the coordination this scheme exists to remove; an ADR
# still being written takes the number of the pull request that adds it, and is
# cited by slug until then.
LEGACY_IN_FLIGHT = {
    18: "content-store-capability-and-binding",  # hosted-presigned-catalogs-fixes
}

# The slug must start with a letter, which is what keeps it lexically disjoint
# from a number. A digit-initial slug would make `ADR-2024-migration` parse as a
# citation of ADR-2024 -- so the grammar enforces the property the two citation
# forms rely on, rather than leaving it to convention.
FILENAME_RE = re.compile(
    rf"^(?P<num>\d{{4,}}|{PLACEHOLDER})-(?P<slug>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)\.md$"
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
#
# The leading `\b` supplies the boundary on the other side: without it, the tail
# of a longer word reads as a citation, so `BADR-0011` would be reported as a
# reference to an ADR nobody mentioned.
REFERENCE_RE = re.compile(
    r"\bADR-(?:(?P<num>\d{4,})(?!-\d)|(?P<short>\d{1,3})(?!\d)|(?P<slug>[a-z][a-z0-9-]*))"
)
# The `\s*` before the target is what adr_rename.py's link rewrite already
# tolerates and preserves. Without it the two disagree about what a link is,
# and `[x]( 0099-nope.md)` is a dead link this never reports.
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(?P<target>[^)#\s]+\.md)[^)]*\)")

# Regions that display an ADR reference rather than make one. A document about
# the numbering scheme necessarily prints numbers that do not resolve, and CI
# output pasted into an ADR would otherwise fail the very check that produced
# it. Stripped before references and links are collected.
CODE_FENCE_RE = re.compile(
    r"^(?P<fence>```|~~~).*?^(?P=fence)", re.MULTILINE | re.DOTALL
)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
URL_RE = re.compile(r"<?https?://[^\s>)]+>?")

# Any Markdown link, target captured, because the two checks want different
# halves of it. The commit check drops the whole thing: stripping only the URL
# would leave the SHA behind as link text, so the linked form it recommends
# would be the one thing that always failed it. The path check keeps the target
# and drops the text, because that is which half is a citation -- a relative
# target names a file in this repository, while the text is prose and may well
# be an upstream path that is not ours to resolve.
MD_LINK_ANY_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)]*)\)")

# A cited repo path: slash-joined segments whose last one carries a file
# extension. The extension is what separates a path from prose -- without it
# `try/except`, `I/O`, `NaN/NaT` and `build/cache` all read as paths, and most
# slash-joined tokens in docs/adr are that kind of alternation rather than a
# path. The cost is that a cited *directory* is never checked, because
# `python/xorq/writes/` and `commit/publish` are the same shape.
#
# The `:NNN` suffix is consumed so it is not read as part of the filename, then
# ignored: line numbers drift on every edit above the line they name.
#
# The lookbehind stops a scan starting part-way along a longer path, which is
# what keeps `s3://bucket/foo.parquet` from reading as `bucket/foo.parquet`.
PATH_CITATION_RE = re.compile(
    r"(?<![\w./-])(?P<path>[\w.-]+(?:/[\w.-]+)*/[\w.-]+\.[A-Za-z]\w*)(?::\d+)?"
)

# A bare short commit: seven to twelve hex characters, neither linked nor
# fenced. Both a digit and an a-f letter are required, which keeps dates
# (`20260427`) and hex-lettered words (`defaced`) out, at the price of the few
# short SHAs that are all digits or all letters.
#
# Twelve is the ceiling because this repository's own content hashes are sixteen
# hex characters and its ADRs are largely about hashing: `38317617c8a70d3a` is a
# build hash, not a commit. A full forty-character SHA is left alone for a
# different reason -- it is unambiguous and greppable, and abbreviation is the
# part that rots.
BARE_SHA_RE = re.compile(
    r"(?<![\w/-])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])(?P<sha>[0-9a-f]{7,12})(?![\w-])"
)


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

        For things that are legitimate now and wrong later: a named reference
        whose ADR is still on an unlanded branch, or an allowlist entry whose
        ADR has landed and which someone else has to delete.
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


def strip_shown_code(text: str) -> str:
    """Drop regions that display an ADR reference rather than make one."""
    for pattern in (CODE_FENCE_RE, INLINE_CODE_RE, URL_RE):
        text = pattern.sub(" ", text)
    return text


def strip_transcripts(text: str) -> str:
    """Drop fenced blocks and URLs, keeping inline code spans.

    Where the path and commit checks part company with `strip_shown_code`. For
    an ADR reference a backtick means "displayed, not cited" -- `` `ADR-9999` ``
    in a document about numbering names no ADR. For a path or a commit it is
    simply how you write one: strip inline spans and there is *nothing* left to
    check, because every path citation in docs/adr is inside backticks.

    A fence still comes out, and is the escape hatch for showing a path that is
    not meant to resolve. A URL's path components belong to another site.
    """
    for pattern in (CODE_FENCE_RE, URL_RE):
        text = pattern.sub(" ", text)
    return text


class TrackedPaths:
    """Every path git tracks, indexed by suffix, built once and only if needed.

    Resolution is by suffix because ADRs cite paths abbreviated for
    readability -- `dasher/_opaque.py` for
    `python/xorq/common/utils/dasher/_opaque.py` -- and about a quarter of the
    citations in docs/adr are that form, so a check wanting full paths would
    either ignore them or demand a sweep of landed ADRs to lengthen them. An
    ambiguous suffix counts as resolved: `expr/api.py` names both the real
    module and the vendored ibis one, and picking between two real files is
    guesswork this check has no business doing.

    The index is what git tracks rather than what is on disk, because "a repo
    path" is the question being asked: `.git/annex/objects` in ADR-0003 was
    never a file here, `docs/_site` is build output, and an untracked local
    script should not make one person's run pass where CI's fails.

    Built on first use, so a directory that cites no paths never shells out to
    git and the guard keeps working outside a repository.
    """

    def __init__(self) -> None:
        self._suffixes: frozenset[str] | None = None
        # Read by `check_directory` to report once, rather than once per ADR.
        self.unavailable = False

    def load(self) -> bool:
        """Read the index, or report False having recorded why."""
        if self._suffixes is not None:
            return True
        if self.unavailable:
            return False
        result = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, text=True
        )
        if result.returncode != 0:
            self.unavailable = True
            return False
        suffixes: set[str] = set()
        for tracked in result.stdout.split("\0"):
            parts = tracked.split("/")
            # Only suffixes that keep a slash. A citation has to contain one --
            # a bare `core.py` would match anything of that name, which makes it
            # both unresolvable and indistinguishable from an illustrative one.
            for start in range(len(parts) - 1):
                suffixes.add("/".join(parts[start:]))
        self._suffixes = frozenset(suffixes)
        return True

    def contains(self, citation: str) -> bool:
        return self._suffixes is not None and citation in self._suffixes


def check_directory(problems: Problems, added: list[Path] | None = None) -> None:
    """Validate every ADR on disk.

    `added` names the ADRs this pull request adds, when a base ref was given. It
    is what the path and commit citation checks read to tell a citation that was
    wrong when it was written from one that has merely aged; without it every
    such citation is a warning, which is what a plain local run sees.
    """
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
                "slug is lowercase words joined by single hyphens and starts "
                "with a letter",
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
    new_names = {path.name for path in added or ()}
    tracked = TrackedPaths()
    for path, number, raw in checkable:
        text = path.read_text(encoding="utf-8")
        is_new = path.name in new_names
        check_heading(problems, path, number, raw, text)
        check_references(problems, path, text, by_number, by_slug)
        check_paths(problems, path, text, tracked, is_new)
        check_shas(problems, path, text, is_new)

    # Once, rather than once per ADR, and a warning rather than an error: an
    # unreadable index means the path check could not run, not that anything is
    # wrong with the ADRs.
    if tracked.unavailable:
        problems.warn(
            ADR_DIR,
            "cited repo paths were not checked: `git ls-files` failed, so there "
            "is nothing to resolve them against. Run from inside a checkout",
        )


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


# Appended to every path and commit report on an ADR already in the tree, and
# the only place this guard tells anyone *not* to fix something. Kept to two
# sentences because `--format github` shows it as an inline annotation; the
# reasoning lives in `check_paths` and docs/adr/README.md.
AGED_CITATION = (
    "If that was accurate when this ADR landed, leave the ADR alone: it records "
    "what was true when the decision was made, and editing a landed ADR to "
    "match today's tree destroys that record. Fix it only if it was never right"
)


def check_paths(
    problems: Problems,
    path: Path,
    text: str,
    tracked: TrackedPaths,
    is_new: bool,
) -> None:
    """Resolve every repo path an ADR cites.

    An unresolved path is an error in an ADR this pull request adds and a
    warning in one that has already landed. The asymmetry is the whole design,
    because the two cases are opposites rather than degrees of the same thing:
    in a new ADR the path is simply wrong, nothing having had time to move,
    while in a landed one it is usually the document doing its job. ADR-0006 and
    ADR-0007 cite `dask_normalize_expr.py`, accurate until #1951 replaced dask
    tokenization; failing on that would pressure people to edit landed ADRs,
    which docs/adr/README.md tells them not to do.

    This is also the answer to the "marked historical block" that #2203 asked
    for. A marker would put the burden on authors to remember it and would let a
    genuinely wrong path be waved through; the pull request diff already knows
    which citations are new. The cost is that a path added to a landed ADR by an
    amendment only warns -- scoping to the added *lines* rather than the added
    *files* would close that, at the price of parsing diff hunks.

    Nothing is reported for a path this run cannot resolve, only for one it can
    prove absent, which is why `load` failing is silent here.
    """
    prose = MD_LINK_ANY_RE.sub(r" \g<target> ", strip_transcripts(text))
    citations = {
        match.group("path")
        for match in PATH_CITATION_RE.finditer(prose)
        # A `..` *segment* is relative to the document rather than the repository
        # root, and this check anchors at the root; same-directory ADR links are
        # already covered by `check_references`. Tested for by segment rather
        # than as a substring so that a citation merely containing dots is
        # reported instead of silently skipped: `...path/to/x.py` is a missing
        # space, and a report quoting the dots is what shows the author that.
        if ".." not in match.group("path").split("/")
    }
    if not citations or not tracked.load():
        return

    for citation in sorted(citations):
        if tracked.contains(citation):
            continue
        message = (
            f"cites `{citation}`, which is not a path git tracks in this repository. "
        )
        if is_new:
            problems.add(
                path,
                message + "An ADR's citations have to resolve when it lands: "
                "check the spelling, or put the path in a fenced block if it "
                "names something this pull request does not create",
            )
        else:
            problems.warn(path, message + AGED_CITATION)


def check_shas(problems: Problems, path: Path, text: str, is_new: bool) -> None:
    """Report commits cited as a bare short SHA.

    A short SHA reads as authoritative and resolves for whoever wrote it, then
    resolves for nobody once the branch it was on is squash-merged or deleted --
    ADR-0007 cites `ce8004bc` as being "on `main`", and it is unreachable in a
    full clone today. Abbreviations also grow ambiguous as a repository does, so
    one that resolves now is not durable either.

    Same error-versus-warning split as `check_paths`, for the same reason.
    """
    prose = MD_LINK_ANY_RE.sub(" ", strip_transcripts(text))
    for sha in sorted({match.group("sha") for match in BARE_SHA_RE.finditer(prose)}):
        message = (
            f"cites the bare short commit `{sha}`. A short SHA is unreachable "
            "once the branch it was on is gone, and abbreviations grow "
            "ambiguous as the repository grows. "
        )
        if is_new:
            problems.add(
                path,
                message + "Cite the pull request that carried the change, or "
                "link the commit -- a linked SHA is not bare, and a fenced "
                "block is not scanned",
            )
        else:
            problems.warn(path, message + AGED_CITATION)


def added_adrs(base: str, pair_renames: bool = False) -> list[Path] | None:
    """ADR files this branch adds relative to the merge base with `base`.

    None means the base ref could not be resolved -- an unfetched sha, a
    shallow clone, or a branch this clone does not have. The message is written
    here; the caller turns it into an exit code.

    `--no-renames` covers the renumber. Rename detection is on by default, and
    it pairs a rename only when the old name is present at the base -- so it
    does not fire for the ordinary flow, where the `XXXX-<slug>.md` draft was
    never on `main` and the ADR arrives as a plain `A`. It fires for the case
    `adr_rename.py` exists to serve second: renumbering an ADR that has already
    landed, where git sees one file move and `--diff-filter=A` would report no
    new ADR at all. Without this flag the branch resolving a collision is the
    one branch the number check never runs on.

    `pair_renames` asks for the opposite, and exactly one caller wants it: the
    citation checks in `check_paths` and `check_shas`, which error only on an ADR
    whose text is new. A renumbered ADR's citations are as historical after the
    renumber as before, so reading the move as an addition would fail the branch
    on prose nobody touched.
    """
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            *(() if pair_renames else ("--no-renames",)),
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


def check_allowlist(problems: Problems, added: list[Path] | None) -> None:
    """Report allowlist entries that have done their job.

    An entry exists to let one legacy-numbered ADR land from a branch that
    predates this scheme. Once that file is in the tree and this run is not the
    one adding it, the exemption is spent.

    A spent entry is not a licence to claim that number: while the ADR is
    present, anything else at that number fails on the duplicate check, and
    `is_allowlisted` matches the slug too, so the exemption only ever covered
    one filename. The risk is that it outlives the file. Delete or renumber
    that ADR and the entry silently readmits its exact filename carrying any
    content at all -- and this check cannot warn about that, because it keys
    on the file being present. Which is the argument for clearing entries
    while they are still merely redundant.

    It cannot be deleted by the pull request that lands the ADR: CI reads the
    merged code, so the entry has to still be there for `check_added` to allow
    the file, and removing it in the same pull request rejects the very ADR it
    was protecting. The cleanup is therefore always a follow-up -- and the
    comment on LEGACY_IN_FLIGHT saying the list "must only ever shrink" names
    no one and nothing enforced it. This is that enforcement.

    A warning rather than an error, because the entry belongs to whoever landed
    the ADR and an unrelated pull request should not be blocked by it. It
    reappears on every run until someone deletes it.
    """
    being_added = {path.name for path in added or ()}
    for number, slug in sorted(LEGACY_IN_FLIGHT.items()):
        path = ADR_DIR / f"{number:04d}-{slug}.md"
        if path.name in being_added or not path.exists():
            continue
        problems.warn(
            path,
            f"has landed, so the LEGACY_IN_FLIGHT entry for {number:04d} in "
            "scripts/adr_check.py has done its job and should be deleted. It "
            "now exempts a file that is already here, and if this ADR is ever "
            "deleted or renumbered it would let that exact filename back in "
            "unchecked, carrying anything",
        )


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

    # `--pr` is only read by the added-ADR checks, and those need a base to know
    # which ADRs are new. Accepting it alone would silently skip the check the
    # caller asked for, which in CI reads exactly like a pull request that
    # passed.
    if args.pr is not None and not args.base:
        parser.error("--pr needs --base: the new-ADR checks diff against a base ref")

    if not ADR_DIR.is_dir():
        sys.stderr.write(f"{ADR_DIR} not found; run from the repository root\n")
        return 2

    problems = Problems(github=args.format == "github")
    # The diff comes first because the directory pass reads it: only the diff can
    # tell a citation that was wrong when written from one that has aged. Two
    # diffs, not one, because the numbering checks and the citation checks want
    # opposite answers about a renumbered ADR -- see `added_adrs`.
    added = None
    written = None
    if args.base:
        # Sequenced, not batched, so an unresolvable base is reported once.
        added = added_adrs(args.base)
        if added is None:
            return 2
        written = added_adrs(args.base, pair_renames=True)
        if written is None:
            return 2
    check_directory(problems, written)
    if added is not None:
        check_added(problems, added, args.pr)
    # Last, and with `added` in hand: an allowlist entry is only spent if this
    # run is not the one using it.
    check_allowlist(problems, added)

    if problems.warnings:
        sys.stderr.write(
            f"\n{problems.warnings} warning(s), which do not fail the run. A "
            "named reference resolves once its ADR lands; a spent allowlist "
            "entry needs deleting; a path or commit cited by a landed ADR may "
            "have aged, and an ADR that has aged is not one to rewrite.\n"
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
