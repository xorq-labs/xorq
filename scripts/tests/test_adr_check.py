"""Tests for the ADR numbering guard.

`scripts/adr_check.py` is what makes the numbering scheme in `docs/adr/README.md`
enforceable rather than advisory, so its own behaviour is worth pinning: a check
that silently stops firing looks exactly like a repository with no problems.

Each test builds a throwaway `docs/adr/` and runs the guard against it, so
nothing here depends on which ADRs happen to exist in the repository today.

    uv run --no-sync pytest scripts/tests

The guard itself stays stdlib-only and `ci-adr.yml` runs it with the runner's
bare `python3`; these tests are not part of that workflow, which is what keeps
it free of an install step.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import adr_check  # noqa: E402  (path set above)
import adr_rename  # noqa: E402


BASE_ADR = "0011-catalog-single-git-remote.md"
BASE_TEXT = "# ADR-0011: Catalog supports a single git remote\n\nBody.\n"


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


def test_allowlisted_legacy_adr_passes(tree: ADRTree) -> None:
    number, slug = next(iter(adr_check.LEGACY_IN_FLIGHT.items()))
    tree.init_repo()
    tree.write(f"{number:04d}-{slug}.md", f"# ADR-{number:04d}: Legacy\n\nBody.\n")
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 0, result.stderr


def test_allowlisted_number_with_a_different_slug_is_rejected(tree: ADRTree) -> None:
    """The allowlist exempts one file, not a number anyone may claim."""
    number = next(iter(adr_check.LEGACY_IN_FLIGHT))
    tree.init_repo()
    tree.write(
        f"{number:04d}-brand-new-unrelated.md",
        f"# ADR-{number:04d}: Unrelated\n\nBody.\n",
    )
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 1
    assert "still in flight" in result.stderr


def test_batch_of_unrelated_adrs_on_allowlisted_numbers_is_rejected(
    tree: ADRTree,
) -> None:
    """The batch exemption must not become a general two-ADR bypass."""
    numbers = list(adr_check.LEGACY_IN_FLIGHT)[:2]
    tree.init_repo()
    for number in numbers:
        tree.write(
            f"{number:04d}-brand-new-{number}.md",
            f"# ADR-{number:04d}: Unrelated\n\nBody.\n",
        )
    tree.commit_all()
    result = tree.check("--base", "HEAD~1", "--pr", "2211")
    assert result.returncode == 1
    assert "only one ADR can be added" in result.stderr


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


def test_environment_is_restored_between_tests() -> None:
    """monkeypatch.chdir must not leak into the rest of the suite."""
    assert Path.cwd() == Path(os.getcwd())
