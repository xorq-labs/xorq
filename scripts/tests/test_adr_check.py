"""Tests for the ADR tooling: the numbering guard and the scripts around it.

`scripts/adr_check.py` is what makes the numbering scheme in `docs/adr/README.md`
enforceable rather than advisory, so its own behaviour is worth pinning -- for
the reason scripts/README.md gives for testing any guard.

Each test builds a throwaway `docs/adr/` and runs the tooling against it, so
nothing here depends on which ADRs happen to exist in the repository today.
`docs/adr/template.md` is the one deliberate exception, read by the
`scaffoldable` fixture so the scaffolding tests run against the real template.

    uv run --no-sync pytest scripts/tests

The guard itself stays stdlib-only and the `adr` job in `ci-adr.yml` runs it
with the runner's bare `python3`. These tests need pytest, so they run as that
workflow's second job, which is what keeps the first free of an install step.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


# Captured at import, before any test can chdir away from it.
STARTING_DIR = Path.cwd()

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import adr_check  # noqa: E402  (path set above)
import adr_index  # noqa: E402
import adr_new  # noqa: E402
import adr_rename  # noqa: E402


BASE_ADR = "0011-catalog-single-git-remote.md"
BASE_TEXT = "# ADR-0011: Catalog supports a single git remote\n\nBody.\n"
REAL_TEMPLATE = REPO_ROOT / "docs" / "adr" / "template.md"


class ADRTree:
    """A throwaway repository root holding one valid ADR to reference."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.adr_dir = root / "docs" / "adr"
        self.adr_dir.mkdir(parents=True)
        self.write(BASE_ADR, BASE_TEXT)

    def write(self, name: str, text: str) -> Path:
        path = self.adr_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "adr_check.py"), *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def commit_all(self, message: str = "wip") -> None:
        self.git("add", "-A")
        self.git("commit", "-qm", message)

    def init_repo(self) -> None:
        self.git("init", "-q", ".")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")
        self.commit_all("base")


@pytest.fixture
def tree(tmp_path: Path) -> ADRTree:
    return ADRTree(tmp_path)


def test_valid_directory_passes(tree: ADRTree) -> None:
    assert tree.check().returncode == 0


def test_missing_adr_dir_is_a_usage_error(tree: ADRTree, tmp_path: Path) -> None:
    for path in tree.adr_dir.iterdir():
        path.unlink()
    tree.adr_dir.rmdir()
    (tmp_path / "docs").rmdir()
    result = tree.check()
    assert result.returncode == 2
    assert "not found" in result.stderr


# --- the two citation forms, and the asymmetry between them ------------------


def test_named_reference_to_a_numbered_adr_resolves(tree: ADRTree) -> None:
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-catalog-single-git-remote.\n")
    result = tree.check()
    assert result.returncode == 0
    assert "warning" not in result.stderr


def test_named_reference_to_an_unnumbered_adr_resolves(tree: ADRTree) -> None:
    """The forward-reference case: cite an ADR before it has a number."""
    tree.write("XXXX-pending-thing.md", "# ADR-XXXX: Pending\n\nBody.\n")
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-pending-thing.\n")
    result = tree.check()
    assert result.returncode == 1  # the placeholder itself, not the reference
    assert "warning" not in result.stderr


def test_unresolved_named_reference_warns_without_failing(tree: ADRTree) -> None:
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-lands-in-a-later-pr.\n")
    result = tree.check()
    assert result.returncode == 0
    assert "warning" in result.stderr
    assert "ADR-lands-in-a-later-pr" in result.stderr


def test_unresolved_numeric_reference_fails(tree: ADRTree) -> None:
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-9999.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "ADR-9999" in result.stderr


def test_hyphenated_date_is_not_a_reference(tree: ADRTree) -> None:
    """ADR-2026-08-08 must not read as a reference to ADR-2026."""
    tree.write("0012-two.md", "# ADR-0012: Two\n\nDated ADR-2026-08-08 scheme.\n")
    result = tree.check()
    assert result.returncode == 0
    assert "ADR-2026" not in result.stderr


def test_the_tail_of_a_longer_word_is_not_a_reference(tree: ADRTree) -> None:
    """`ADR-` has to start a word, or the guard invents citations nobody made."""
    tree.write("0012-two.md", "# ADR-0012: Two\n\nThe BADR-9999 register.\n")
    assert tree.check().returncode == 0


def test_short_form_citation_is_reported(tree: ADRTree) -> None:
    """ADR-11 is not a valid citation and must not pass silently."""
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-11.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "four digits" in result.stderr


# --- displaying a reference rather than making one ---------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("```\nADR-9999\n```\n", id="fenced-block"),
        pytest.param("~~~\nADR-9999\n~~~\n", id="tilde-fence"),
        pytest.param("Write `ADR-9999` here.\n", id="inline-code"),
        pytest.param("<https://example.com/ADR-9999>\n", id="autolink"),
        pytest.param("```\n[x](0099-nope.md)\n```\n", id="dead-link-in-fence"),
    ],
)
def test_shown_reference_is_not_checked(tree: ADRTree, body: str) -> None:
    tree.write("0012-two.md", f"# ADR-0012: Two\n\n{body}")
    assert tree.check().returncode == 0


def test_a_live_reference_outside_a_fence_still_fails(tree: ADRTree) -> None:
    """The escape hatch must not swallow real references."""
    tree.write(
        "0012-two.md", "# ADR-0012: Two\n\n```\nADR-9999\n```\n\nBut see ADR-8888.\n"
    )
    result = tree.check()
    assert result.returncode == 1
    assert "ADR-8888" in result.stderr
    assert "ADR-9999" not in result.stderr


# --- collisions --------------------------------------------------------------


def test_duplicate_number_is_an_error(tree: ADRTree) -> None:
    tree.write("0011-other-slug.md", "# ADR-0011: Clash\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "already used by" in result.stderr


def test_duplicate_slug_is_an_error(tree: ADRTree) -> None:
    tree.write("0012-catalog-single-git-remote.md", "# ADR-0012: Dup\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "slug `catalog-single-git-remote` is already used" in result.stderr


def test_a_duplicate_file_is_still_checked_for_other_problems(tree: ADRTree) -> None:
    """One run reports every problem, including on the duplicate itself."""
    tree.write(
        "0011-second-file.md",
        "# ADR-0011: Clash\n\nSee ADR-8888 and [x](0099-nope.md).\n",
    )
    result = tree.check()
    assert result.returncode == 1
    assert "already used by" in result.stderr
    assert "ADR-8888" in result.stderr
    assert "0099-nope.md" in result.stderr


# --- unnumbered ADRs are checked like any other ------------------------------


def test_placeholder_is_reported(tree: ADRTree) -> None:
    tree.write("XXXX-pending-thing.md", "# ADR-XXXX: Pending\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "placeholder" in result.stderr


def test_bad_references_inside_a_placeholder_are_reported(tree: ADRTree) -> None:
    tree.write(
        "XXXX-pending-thing.md",
        "# ADR-XXXX: Pending\n\nSee ADR-9999 and [x](0099-nope.md).\n",
    )
    result = tree.check()
    assert result.returncode == 1
    assert "ADR-9999" in result.stderr
    assert "0099-nope.md" in result.stderr


def test_placeholder_heading_must_say_placeholder(tree: ADRTree) -> None:
    tree.write("XXXX-pending-thing.md", "# ADR-0007: Wrong\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "heading says ADR-0007" in result.stderr


# --- filenames ---------------------------------------------------------------


def test_a_digit_initial_slug_is_rejected(tree: ADRTree) -> None:
    """The invariant the two citation forms rest on, enforced by the grammar.

    A slug starting with a digit is not lexically disjoint from a number:
    `ADR-2024-migration` parses as a citation of ADR-2024, so the filename must
    not be able to create one.
    """
    tree.write("2211-2024-migration.md", "# ADR-2211: Migration\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "starts with a letter" in result.stderr


def test_malformed_filename_is_reported(tree: ADRTree) -> None:
    tree.write("0012_Bad_Slug.md", "# ADR-0012: Bad\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "does not match" in result.stderr


def test_heading_mismatch_quotes_the_filename_as_written(tree: ADRTree) -> None:
    """A five-digit prefix is echoed back literally, not reformatted to four."""
    tree.write("00011-padded-thing.md", "# ADR-0011: Padded\n\nBody.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "the filename says 00011" in result.stderr


def test_a_link_with_a_leading_space_is_still_checked(tree: ADRTree) -> None:
    """The guard and the sweep have to agree on what a link is.

    adr_rename.py's rewrite tolerates the leading space and puts it back, so a
    link the sweep would repoint must be one the guard would report.
    """
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee [it]( 0099-nope.md).\n")
    result = tree.check()
    assert result.returncode == 1
    assert "0099-nope.md, which does not exist" in result.stderr


def test_adr_in_a_subdirectory_is_reported(tree: ADRTree) -> None:
    tree.write("sub/2211-hidden.md", "no heading\n\nSee ADR-8888.\n")
    result = tree.check()
    assert result.returncode == 1
    assert "subdirectories" in result.stderr


# --- the --base checks, which run against a diff -----------------------------


def test_correct_pull_request_number_passes(tree: ADRTree) -> None:
    tree.init_repo()
    tree.write("2211-new-decision.md", "# ADR-2211: New\n\nBody.\n")
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 0, result.stderr


def test_wrong_pull_request_number_fails(tree: ADRTree) -> None:
    tree.init_repo()
    tree.write("2211-new-decision.md", "# ADR-2211: New\n\nBody.\n")
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2212")
    assert result.returncode == 1
    assert "must equal the pull request number" in result.stderr


def test_a_renumbered_adr_is_still_a_new_adr(tree: ADRTree) -> None:
    """Rename detection hides the renumber, which is the collision fix.

    Git pairs a rename only when the old name is at the base, so the ordinary
    flow is unaffected -- the draft was never on `main`. Renumbering a landed
    ADR is the case that pairs: one file moves, and `--diff-filter=A` alone
    reports no new ADR, leaving the number unchecked on precisely the branch
    that exists to fix a number.
    """
    tree.init_repo()
    tree.write("0016-old-thing.md", "# ADR-0016: Old\n\nBody enough to pair.\n")
    tree.commit_all()
    tree.git("mv", "docs/adr/0016-old-thing.md", "docs/adr/0019-old-thing.md")
    tree.write("0019-old-thing.md", "# ADR-0019: Old\n\nBody enough to pair.\n")
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 1
    assert "at least 1000" in result.stderr


def test_new_legacy_number_is_rejected(tree: ADRTree) -> None:
    tree.init_repo()
    tree.write("0026-sneaking-in-sequential.md", "# ADR-0026: Nope\n\nBody.\n")
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 1
    assert "at least 1000" in result.stderr


def test_two_new_adrs_in_one_pull_request_is_rejected(tree: ADRTree) -> None:
    tree.init_repo()
    tree.write("2211-one.md", "# ADR-2211: One\n\nBody.\n")
    tree.write("2211-two.md", "# ADR-2211: Two\n\nBody.\n")
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 1
    assert "only one ADR can be added" in result.stderr
    # Splitting is only cheap if the author knows the split ADRs can still cite
    # each other; without that the rule reads as "reserve a number or wait".
    assert "ADR-<slug>" in result.stderr


@pytest.fixture
def fake_allowlist(monkeypatch: pytest.MonkeyPatch) -> dict[int, str]:
    """A synthetic LEGACY_IN_FLIGHT, so these tests outlive the real one.

    Reading the live mapping made each of these need an entry to exist. That
    is policy data on someone else's branch: it shrank from four to one when
    #2200 landed and is documented as shrinking to nothing, at which point
    every test that read it fails for a reason unrelated to the code.
    """
    fake = {18: "one-legacy-adr", 19: "another-legacy-adr"}
    monkeypatch.setattr(adr_check, "LEGACY_IN_FLIGHT", fake)
    return fake


def test_allowlisted_legacy_adr_passes(fake_allowlist: dict[int, str]) -> None:
    problems = adr_check.Problems()
    adr_check.check_added(problems, [Path("docs/adr/0018-one-legacy-adr.md")], 2211)
    assert problems.count == 0


def test_allowlisted_number_with_a_different_slug_is_rejected(
    fake_allowlist: dict[int, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The allowlist exempts one file, not a number anyone may claim."""
    problems = adr_check.Problems()
    adr_check.check_added(problems, [Path("docs/adr/0018-brand-new.md")], 2211)
    assert problems.count == 1
    assert "still in flight" in capsys.readouterr().err


def test_an_allowlist_entry_is_reported_once_its_adr_has_landed(
    tree: ADRTree,
    fake_allowlist: dict[int, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The entry cannot be deleted by the pull request that lands the ADR.

    CI reads the merged code, so removing it there rejects the very file it
    was protecting -- the cleanup is always a follow-up, and a follow-up
    nobody is reminded of is one that does not happen. What makes it worth
    reminding about is that the exemption outlives the file: see
    `test_a_spent_entry_readmits_its_filename_once_the_adr_is_gone`.
    """
    tree.write("0018-one-legacy-adr.md", "# ADR-0018: Legacy\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    problems = adr_check.Problems()
    adr_check.check_allowlist(problems, None)
    assert problems.warnings == 1
    # A warning, not an error: the entry belongs to whoever landed the ADR,
    # and someone else's pull request should not be blocked by it.
    assert problems.count == 0
    assert "has done its job" in capsys.readouterr().err


def test_an_allowlist_entry_whose_adr_is_absent_is_not_reported(
    tree: ADRTree, fake_allowlist: dict[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary state of an entry: its branch has not landed yet."""
    monkeypatch.chdir(tree.root)
    problems = adr_check.Problems()
    adr_check.check_allowlist(problems, None)
    assert problems.warnings == 0


def test_an_entry_this_run_is_using_is_not_reported(
    tree: ADRTree, fake_allowlist: dict[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The landing pull request: in use, not spent."""
    tree.write("0018-one-legacy-adr.md", "# ADR-0018: Legacy\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    problems = adr_check.Problems()
    adr_check.check_allowlist(problems, [Path("docs/adr/0018-one-legacy-adr.md")])
    assert problems.warnings == 0


def test_a_spent_entry_is_reported_on_an_unrelated_pull_request(
    tree: ADRTree, fake_allowlist: dict[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape every pull request sees after the ADR lands.

    The push-to-main run reports it once; this is the nag that keeps arriving
    until someone acts, and it arrives through the `--base` path rather than
    the directory-only one.
    """
    tree.write("0018-one-legacy-adr.md", "# ADR-0018: Legacy\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    problems = adr_check.Problems()
    adr_check.check_allowlist(problems, [Path("docs/adr/2222-unrelated.md")])
    assert problems.warnings == 1


def test_a_spent_entry_readmits_its_filename_once_the_adr_is_gone(
    tree: ADRTree, fake_allowlist: dict[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why a spent entry is worth deleting, and why the warning cannot wait.

    While the ADR is present the entry is merely redundant: anything else at
    that number fails the duplicate check. Delete the ADR and the entry
    readmits its exact filename carrying anything -- and by then this check is
    silent, because it keys on the file being present. The warning exists to
    be acted on while the entry is still harmless.
    """
    monkeypatch.chdir(tree.root)
    readmitted = Path("docs/adr/0018-one-legacy-adr.md")

    # The ADR landed and was later removed, so nothing warns any more.
    silent = adr_check.Problems()
    adr_check.check_allowlist(silent, None)
    assert silent.warnings == 0

    # And the entry lets that exact filename back in, carrying anything.
    problems = adr_check.Problems()
    adr_check.check_added(problems, [readmitted], 2222)
    assert problems.count == 0


@pytest.mark.skipif(
    not adr_check.LEGACY_IN_FLIGHT,
    reason="nothing left to exempt, so the wiring cannot be exercised end to end",
)
def test_the_allowlist_check_is_wired_into_the_command(tree: ADRTree) -> None:
    """The one test that reads the live mapping, and only to prove the wiring.

    Everything above calls `check_allowlist` directly, which would not notice
    the call being dropped from `main` or moved inside `if args.base`. This
    runs the command as CI does. It skips once the allowlist is empty, because
    at that point there is no entry to make it speak.
    """
    number, slug = next(iter(adr_check.LEGACY_IN_FLIGHT.items()))
    tree.write(f"{number:04d}-{slug}.md", f"# ADR-{number:04d}: Legacy\n\nBody.\n")
    result = tree.check()
    assert "has done its job" in result.stderr
    assert result.returncode == 0, result.stderr


def test_batch_of_unrelated_adrs_on_allowlisted_numbers_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The batch exemption must not become a general two-ADR bypass.

    The allowlist is synthetic here, and deliberately: this is a fact about
    `check_added`, not about whichever entries happen to be in flight today.
    Reading the live mapping made the test need two entries to exist, so
    shrinking the list to one broke it -- and the list is documented as
    shrinking to nothing.
    """
    monkeypatch.setattr(
        adr_check,
        "LEGACY_IN_FLIGHT",
        {18: "one-legacy-adr", 19: "another-legacy-adr"},
    )
    problems = adr_check.Problems()
    adr_check.check_added(
        problems,
        [Path("docs/adr/0018-brand-new.md"), Path("docs/adr/0019-brand-new.md")],
        2211,
    )
    assert problems.count == 1
    assert "only one ADR can be added" in capsys.readouterr().err


def test_unresolvable_base_is_a_clean_usage_error(tree: ADRTree) -> None:
    tree.init_repo()
    result = tree.check("--base", "no-such-ref", "--pr", "2211")
    assert result.returncode == 2
    assert "cannot diff against" in result.stderr
    assert "Traceback" not in result.stderr


# --- output format -----------------------------------------------------------


def test_annotations_are_off_by_default(tree: ADRTree) -> None:
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-9999.\n")
    assert "::error" not in tree.check().stdout


def test_github_format_emits_annotations(tree: ADRTree) -> None:
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-9999.\n")
    assert "::error file=" in tree.check("--format", "github").stdout


def test_github_format_annotates_warnings_too(tree: ADRTree) -> None:
    """An unresolved slug does not fail the run, so the annotation is the only
    place a reviewer will see it."""
    tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-not-landed-yet.\n")
    result = tree.check("--format", "github")
    assert "::warning file=" in result.stdout
    assert result.returncode == 0


def test_pr_without_a_base_is_a_usage_error(tree: ADRTree) -> None:
    """`--pr` alone reads as a pull request that passed the new-ADR checks,
    when in fact they never ran."""
    result = tree.check("--pr", "2211")
    assert result.returncode == 2
    assert "--base" in result.stderr


# --- the citation sweep in adr_rename.py -------------------------------------


def test_sweep_rewrites_a_prose_citation(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.init_repo()
    target = tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-my-decision.\n")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211")
    assert "ADR-2211" in target.read_text()


def test_sweep_leaves_a_longer_slug_sharing_the_prefix_alone(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.init_repo()
    target = tree.write(
        "0012-two.md",
        "# ADR-0012: Two\n\nSee ADR-tee-node and ADR-tee-node-deferred-writes.\n",
    )
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("tee-node", "2211")
    assert "ADR-2211 and ADR-tee-node-deferred-writes" in target.read_text()


def test_sweep_rewrites_a_relative_link_to_the_placeholder(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the rename leaves a link to a file that no longer exists."""
    tree.init_repo()
    target = tree.write(
        "0012-two.md",
        "# ADR-0012: Two\n\nSee [ADR-my-decision](XXXX-my-decision.md).\n",
    )
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211")
    assert "[ADR-2211](2211-my-decision.md)" in target.read_text()


def test_rename_end_to_end_leaves_a_clean_tree(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rename, heading fix, and sweep together must satisfy the guard."""
    # The citing ADR is already on the base commit; this pull request adds only
    # the new one, which is the shape the one-per-pull-request rule expects.
    tree.write(
        "0012-two.md",
        "# ADR-0012: Two\n\nSee ADR-my-decision at [it](XXXX-my-decision.md).\n",
    )
    tree.init_repo()
    tree.write("XXXX-my-decision.md", "# ADR-XXXX: Mine\n\nBody.\n")
    tree.commit_all()
    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(sys, "argv", ["adr_rename.py"])

    # Stand in for `gh pr view`, which needs a real forge.
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "HEAD~1"))
    assert adr_rename.main() == 0

    assert not (tree.adr_dir / "XXXX-my-decision.md").exists()
    assert (tree.adr_dir / "2211-my-decision.md").exists()
    body = (tree.adr_dir / "0012-two.md").read_text()
    assert "See ADR-2211 at [it](2211-my-decision.md)." in body


# --- sweeping the number an ADR is moving away from ---------------------------
#
# A missed slug citation still resolves and a dead link is reported, but a
# missed *number* resolves silently to whichever ADR later holds it -- and in
# code, which the guard never reads, nothing reports it at all. That is the
# failure these pin.


def test_sweep_rewrites_the_previous_number(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.init_repo()
    target = tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-0017.\n")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211", "0017")
    assert "See ADR-2211." in target.read_text()


def test_sweep_rewrites_the_short_form_of_the_previous_number(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ADR-17` is the typo the guard reports; renumbering should fix it too."""
    tree.init_repo()
    target = tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-17.\n")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211", "0017")
    assert "See ADR-2211." in target.read_text()


def test_sweep_never_rewrites_the_placeholder(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-XXXX identifies nothing, so no sweep may claim it.

    Every draft in flight carries it, and so does template.md permanently. A
    repository-wide rewrite would renumber all of them to whichever ADR
    happened to be renamed -- silently, in template.md's case, which the guard
    exempts from the reference checks.
    """
    tree.init_repo()
    other = tree.write("XXXX-other-draft.md", "# ADR-XXXX: Other\n\nBody.\n")
    template = tree.write("template.md", "# ADR-XXXX: <title>\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211")
    assert other.read_text().startswith("# ADR-XXXX:")
    assert template.read_text().startswith("# ADR-XXXX:")


def test_rename_rewrites_the_placeholder_inside_the_file_it_numbers(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where ADR-XXXX does mean one ADR: its own body, citing itself."""
    tree.init_repo()
    tree.write(
        "XXXX-my-decision.md",
        "# ADR-XXXX: Mine\n\nSupersedes nothing; ADR-XXXX stands alone.\n",
    )
    tree.commit_all()
    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(sys, "argv", ["adr_rename.py"])
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "HEAD~1"))

    assert adr_rename.main() == 0

    body = (tree.adr_dir / "2211-my-decision.md").read_text()
    assert body.startswith("# ADR-2211:")
    assert "ADR-2211 stands alone" in body


@pytest.mark.parametrize(
    ("name", "body", "expected"),
    [
        pytest.param(
            "0012-two.md",
            "# ADR-0012: Two\n\nCite it as `ADR-my-decision` before it lands.\n",
            "`ADR-my-decision`",
            id="markdown-inline-code",
        ),
        pytest.param(
            "0012-two.md",
            "# ADR-0012: Two\n\n```\nADR-my-decision\n```\n",
            "```\nADR-my-decision\n```",
            id="markdown-fence",
        ),
        pytest.param(
            "guide.qmd",
            "# Guide\n\nCite it as `ADR-my-decision` before it lands.\n",
            "`ADR-my-decision`",
            id="quarto-inline-code",
        ),
        pytest.param(
            "notes.py",
            '"""Mechanism. See `ADR-my-decision`."""\n',
            "`ADR-2211`",
            id="code-is-swept-throughout",
        ),
    ],
)
def test_sweep_honours_the_boundary_the_guard_reads_by(
    tree: ADRTree,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    body: str,
    expected: str,
) -> None:
    """Prose that displays a citation is documentation, not a citation.

    docs/adr/README.md and CONTRIBUTING.md both teach the named form using a
    real-looking slug; numbering the ADR one of them names must not edit the
    page that explains the convention. That has to hold for Quarto too, since
    docs/ is written in it. A backtick in a docstring makes no such claim, and
    a stale number in code is the failure the sweep exists for, so code is
    rewritten throughout.
    """
    tree.init_repo()
    target = tree.adr_dir / name if name.endswith(".md") else tree.root / name
    target.write_text(body, encoding="utf-8")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211")
    assert expected in target.read_text()


def test_sweep_reaches_code_not_only_adrs(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #2129 lesson: most citations of a number live in docstrings."""
    tree.init_repo()
    module = tree.root / "python" / "xorq" / "writes" / "publish.py"
    module.parent.mkdir(parents=True)
    module.write_text('"""Publish a changeset. See ADR-0017."""\n', encoding="utf-8")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211", "0017")
    assert "See ADR-2211." in module.read_text()


def test_sweep_leaves_a_hyphenated_date_alone(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the guard's own guard: ADR-2026-08-10 is a date, not ADR-2026."""
    tree.init_repo()
    target = tree.write("0012-two.md", "# ADR-0012: Two\n\nOn ADR-2026-08-10.\n")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211", "2026")
    assert "On ADR-2026-08-10." in target.read_text()


def test_sweep_leaves_a_neighbouring_number_alone(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.init_repo()
    target = tree.write(
        "0012-two.md", "# ADR-0012: Two\n\nSee ADR-0017 and ADR-0170.\n"
    )
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211", "0017")
    assert "See ADR-2211 and ADR-0170." in target.read_text()


def test_sweep_can_be_told_to_leave_the_previous_number(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.init_repo()
    target = tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-0017.\n")
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211", "0017", include_previous=False)
    assert "See ADR-0017." in target.read_text()


def test_rerunning_sweeps_again_rather_than_declaring_victory(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename and the sweep are separate steps; only the first shows.

    A run that stopped between them leaves the right filename and stale
    citations, so a rerun that reported "already carries number" and stopped
    would make the half-finished state unrecoverable by the tool that caused
    it.
    """
    tree.init_repo()
    tree.write("2211-my-decision.md", "# ADR-2211: Mine\n\nBody.\n")
    missed = tree.write("0012-two.md", "# ADR-0012: Two\n\nSee ADR-my-decision.\n")
    tree.commit_all()
    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(sys, "argv", ["adr_rename.py", "my-decision"])
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "HEAD~1"))

    assert adr_rename.main() == 0
    assert "See ADR-2211." in missed.read_text()


def test_a_failed_search_is_not_reported_as_nothing_to_do(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """git grep exits 1 for "no matches" and 128 for "could not look".

    Reading those alike is the worst failure this tool has: it renames the
    file, announces that there was nothing to rewrite, and exits 0 with every
    citation left pointing at a name that no longer exists.
    """
    # No init_repo, so `git grep` has no repository to search.
    tree.write("XXXX-my-decision.md", "# ADR-XXXX: Mine\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    assert adr_rename.sweep_references("my-decision", "2211") is None

    monkeypatch.setattr(sys, "argv", ["adr_rename.py"])
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "main"))
    assert adr_rename.main() == 1
    # The rename itself stands; undoing it would surprise more than it helps.
    assert (tree.adr_dir / "2211-my-decision.md").exists()
    assert "stale" in capsys.readouterr().err


def test_sweep_keeps_a_link_title_and_its_spacing(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewrite replaces the filename, not the whole link."""
    tree.init_repo()
    target = tree.write(
        "0012-two.md",
        '# ADR-0012: Two\n\nSee [it]( XXXX-my-decision.md "Why").\n',
    )
    monkeypatch.chdir(tree.root)
    adr_rename.sweep_references("my-decision", "2211")
    assert '[it]( 2211-my-decision.md "Why")' in target.read_text()


def test_rename_moves_a_draft_that_was_never_committed(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git mv` exits 128 on an untracked file, and adr_new.py leaves one.

    Renaming before committing the draft is an easy first mistake, and it used
    to end in a traceback from `check=True`.
    """
    tree.init_repo()
    tree.write("XXXX-my-decision.md", "# ADR-XXXX: Mine\n\nBody.\n")  # not committed
    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(sys, "argv", ["adr_rename.py"])
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "HEAD"))

    assert adr_rename.main() == 0
    assert (tree.adr_dir / "2211-my-decision.md").read_text().startswith("# ADR-2211:")


def test_rename_refuses_to_land_on_an_existing_file(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Overwriting the ADR already at that number would lose it silently."""
    tree.init_repo()
    tree.write("XXXX-my-decision.md", "# ADR-XXXX: Mine\n\nBody.\n")
    landed = tree.write("2211-my-decision.md", "# ADR-2211: Landed\n\nKeep me.\n")
    tree.commit_all()
    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(sys, "argv", ["adr_rename.py"])
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "HEAD~1"))

    assert adr_rename.main() == 1
    assert "Keep me." in landed.read_text()
    assert "already exists" in capsys.readouterr().err


def test_a_missing_gh_is_reported_not_raised(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """gh is the only way to the number, but it is not a dependency of the
    lightest contribution there is."""

    def no_gh(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("gh")

    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(adr_rename.subprocess, "run", no_gh)
    assert adr_rename.pull_request() is None
    assert "gh" in capsys.readouterr().err


def test_claimant_reports_an_adr_still_holding_the_number(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ambiguity case: citations of 0011 could mean either ADR, so the
    rename must not guess which."""
    tree.write("0011-mine.md", "# ADR-0011: Mine\n\nBody.\n")
    # ADR_DIR is relative, so this must run from the throwaway root or it
    # inspects the repository's own ADRs.
    monkeypatch.chdir(tree.root)
    moving = adr_rename.ADR_DIR / "0011-mine.md"
    assert adr_rename.claimant("0011", moving) == adr_rename.ADR_DIR / BASE_ADR
    assert adr_rename.claimant("0099", moving) is None
    assert adr_rename.claimant("XXXX", moving) is None


def test_pick_source_finds_a_numbered_adr_by_slug(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.write("0017-my-decision.md", "# ADR-0017: Mine\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    assert adr_rename.pick_source("my-decision") == Path("docs/adr/0017-my-decision.md")


def test_a_bare_run_never_renumbers_a_landed_adr(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renumbering is destructive enough to require naming the ADR: with no
    argument and no placeholder, there is nothing to do."""
    tree.write("0017-my-decision.md", "# ADR-0017: Mine\n\nBody.\n")
    monkeypatch.chdir(tree.root)
    assert adr_rename.pick_source(None) is None


def test_renumber_end_to_end_moves_the_number_out_of_code(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #2129 shape: an ADR holding a number that main has since used."""
    tree.init_repo()
    tree.write("0017-my-decision.md", "# ADR-0017: Mine\n\nBody.\n")
    module = tree.root / "backend.py"
    module.write_text('"""Mechanism (ADR-0017)."""\n', encoding="utf-8")
    tree.commit_all()
    monkeypatch.chdir(tree.root)
    monkeypatch.setattr(sys, "argv", ["adr_rename.py", "my-decision"])
    monkeypatch.setattr(adr_rename, "pull_request", lambda: ("2211", "HEAD~1"))

    assert adr_rename.main() == 0

    assert not (tree.adr_dir / "0017-my-decision.md").exists()
    renamed = tree.adr_dir / "2211-my-decision.md"
    assert renamed.read_text().startswith("# ADR-2211:")
    assert "(ADR-2211)" in module.read_text()


# --- scaffolding in adr_new.py -----------------------------------------------


@pytest.fixture
def scaffoldable(tree: ADRTree, monkeypatch: pytest.MonkeyPatch) -> ADRTree:
    """A tree with the real template in place, ready for `adr_new.main`."""
    tree.write("template.md", REAL_TEMPLATE.read_text(encoding="utf-8"))
    monkeypatch.chdir(tree.root)
    return tree


def scaffold(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["adr_new.py", *argv])
    return adr_new.main()


def test_scaffold_writes_the_placeholder_filename(
    scaffoldable: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert scaffold(monkeypatch, "my-decision") == 0
    written = scaffoldable.adr_dir / "XXXX-my-decision.md"
    assert written.read_text(encoding="utf-8").startswith("# ADR-XXXX:")


@pytest.mark.parametrize(
    "slug",
    [
        pytest.param("My-Decision", id="uppercase"),
        pytest.param("my_decision", id="underscore"),
        pytest.param("my--decision", id="doubled-hyphen"),
        pytest.param("-my-decision", id="leading-hyphen"),
        pytest.param("my-decision-", id="trailing-hyphen"),
        pytest.param("", id="empty"),
        # A digit-initial slug is what would break the two citation forms
        # apart: `ADR-2024-migration` reads as a citation of ADR-2024.
        pytest.param("2024-migration", id="digit-initial"),
    ],
)
def test_scaffold_rejects_a_slug_the_guard_would_reject(
    scaffoldable: ADRTree, monkeypatch: pytest.MonkeyPatch, slug: str
) -> None:
    """Anything FILENAME_RE would not match must fail here, not in CI."""
    assert scaffold(monkeypatch, slug) == 1
    assert list(scaffoldable.adr_dir.glob("XXXX-*.md")) == []


def test_scaffold_refuses_to_overwrite(
    scaffoldable: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = scaffoldable.write("XXXX-my-decision.md", "# ADR-XXXX: Mine\n\nDraft.\n")
    assert scaffold(monkeypatch, "my-decision") == 1
    assert "Draft." in existing.read_text(encoding="utf-8")


def test_scaffold_without_a_slug_is_a_usage_error(
    scaffoldable: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert scaffold(monkeypatch) == 2
    assert scaffold(monkeypatch, "one", "two") == 2


def test_scaffold_outside_the_repository_root_says_so(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No template means the caller is in the wrong directory, not that the
    slug was bad -- exit 2, not 1."""
    monkeypatch.chdir(tree.root)
    assert scaffold(monkeypatch, "my-decision") == 2


def test_a_freshly_scaffolded_adr_satisfies_the_guard(
    scaffoldable: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The template is exempt from the reference checks; a copy of it is not.

    So a citation in the template that stops resolving would send every new ADR
    into CI already failing, and nothing else would report it. The placeholder
    itself is expected to fail until `adr_rename.py` runs.
    """
    assert scaffold(monkeypatch, "my-decision") == 0
    result = scaffoldable.check()
    assert "which does not exist" not in result.stderr
    assert "is not in this directory" not in result.stderr
    assert "placeholder" in result.stderr


def test_adr_new_agrees_with_the_guard_on_the_shared_constants() -> None:
    """adr_new stays import-free of the guard so scaffolding works even with a
    broken one, which leaves these two declarations to keep in step by hand."""
    assert adr_new.ADR_DIR == adr_check.ADR_DIR
    assert adr_new.PLACEHOLDER == adr_check.PLACEHOLDER


# --- the generated index in adr_index.py -------------------------------------
#
# The index is printed, never stored, so nothing here guards a committed file.
# What is worth pinning is that it reads the ADRs correctly and orders them
# numerically -- the one thing `ls` cannot do once numbers reach five digits.


def test_index_links_the_number_and_prints_the_title(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status is the next two tests; this one is the number and the heading."""
    monkeypatch.chdir(tree.root)
    table = adr_index.render(adr_index.entries())
    assert "[0011](0011-catalog-single-git-remote.md)" in table
    assert "Catalog supports a single git remote" in table


def test_index_shows_status_as_text_not_a_link(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table of relative links reads worse than the sentence it came from."""
    tree.write(
        "0012-two.md",
        "# ADR-0012: Two\n\n- **Status:** Superseded by [ADR-0011](0011-catalog-single-git-remote.md)\n",
    )
    monkeypatch.chdir(tree.root)
    table = adr_index.render(adr_index.entries())
    assert "| Superseded by ADR-0011 |" in table


def test_index_sorts_numerically_not_lexically(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason this tool exists: `10000-x.md` sorts before `2129-x.md`.

    Numbers are not zero-padded, so ordering cannot come from the filename.
    """
    tree.write("2129-mid.md", "# ADR-2129: Mid\n\n- **Status:** Accepted\n")
    tree.write("10000-late.md", "# ADR-10000: Late\n\n- **Status:** Accepted\n")
    monkeypatch.chdir(tree.root)
    numbers = [row[0] for row in adr_index.entries()]
    assert numbers == [11, 2129, 10000]


def test_index_puts_an_unnumbered_adr_last(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It has no place in a numeric sequence; first would imply it precedes 0002."""
    tree.write("XXXX-pending.md", "# ADR-XXXX: Pending\n\n- **Status:** Proposed\n")
    monkeypatch.chdir(tree.root)
    rows = adr_index.entries()
    assert rows[-1][0] is None
    assert "[XXXX](XXXX-pending.md)" in adr_index.render(rows)


def test_index_tolerates_a_missing_status(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.write("0012-two.md", "# ADR-0012: Two\n\nNo status line.\n")
    monkeypatch.chdir(tree.root)
    assert "| — |" in adr_index.render(adr_index.entries())


def test_index_skips_a_malformed_filename(
    tree: ADRTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """adr_check.py reports the name precisely; the index just stays runnable."""
    tree.write("not-an-adr.md", "# Nope\n")
    monkeypatch.chdir(tree.root)
    slugs = [row[2] for row in adr_index.entries()]
    assert slugs == ["catalog-single-git-remote"]


def test_environment_is_restored_between_tests() -> None:
    """monkeypatch.chdir must not leak into the rest of the suite.

    Against `os.getcwd()` this would pass however far the suite had wandered,
    since both sides move together. The directory pytest started in is the only
    fixed point that can detect a leak at all.
    """
    assert Path.cwd() == STARTING_DIR
