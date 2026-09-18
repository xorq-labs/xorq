"""Positive controls for the style gate.

`xorq-check-style` exits 0 with no output both when a file is clean and when it
never examined that file, so a rule that has stopped working looks exactly like
a rule with nothing to report. That ambiguity is not hypothetical: before
33b94478 set `src-roots`, `unlisted-import` sat on the enforced list resolving
no modules at all; it now sits on the ratchet with a real count. A directory
argument (xorq-labs/xorq-style#30) still reports a clean tree it never opened.

Every rule therefore owns a fixture it is required to flag. A rule that stops
firing fails here rather than reporting a comfortable zero, and a rule that
arrives in a dependency bump fails here for want of a fixture rather than
joining the enforced set unannounced.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# `--import-mode=importlib`, which ci-test.yml passes, imports this module
# without putting its directory on `sys.path`, so the sibling below is
# unimportable by bare name. Same insert, same reason, as scripts/tests/.
sys.path.insert(0, str(Path(__file__).parent))

from fixtures import FIXTURES, SUPPORT  # noqa: E402  (path set above)


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-lint.yml"
DIFF_GATE = REPO_ROOT / "scripts" / "check-style-diff.sh"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"

# A rule id as the two --disable lists spell it, and as `--list` prints it.
RULE = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["xorq-check-style", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _violations(proc: subprocess.CompletedProcess[str]) -> frozenset[str]:
    return frozenset(v["rule"] for v in json.loads(proc.stdout or "[]"))


@pytest.fixture(scope="module")
def known_rules() -> frozenset[str]:
    """Every rule the installed checker implements."""
    listing = _run("--list", cwd=REPO_ROOT).stdout
    found = frozenset(
        match.group(1)
        for line in listing.splitlines()
        if (match := re.match(rf"\s+({RULE})\s{{2,}}\S", line))
    )
    assert found, "xorq-check-style --list produced no rules"
    return found


@pytest.fixture(scope="module")
def whole_repo_ratchet() -> frozenset[str]:
    """The --disable list the whole-repo gate passes, read from the workflow."""
    lists = re.findall(
        rf"^\s+({RULE}(?:,{RULE})+)$", WORKFLOW.read_text(), re.MULTILINE
    )
    assert len(lists) == 1, (
        f"expected one --disable list in {WORKFLOW.name}, got {len(lists)}"
    )
    return frozenset(lists[0].split(","))


@pytest.fixture(scope="module")
def changed_lines_ratchet() -> frozenset[str]:
    """The disable list the shared changed-lines gate applies."""
    match = re.search(
        rf"^disable=({RULE}(?:,{RULE})+)$", DIFF_GATE.read_text(), re.MULTILINE
    )
    assert match, f"no disable= assignment in {DIFF_GATE.name}"
    return frozenset(match.group(1).split(","))


@pytest.fixture(scope="module")
def repo_violations() -> frozenset[str]:
    """Every rule with at least one violation in the repository as it stands."""
    paths = _lint_paths()
    globs = [f"{path}/*.py" for path in paths]
    # `-z`: NUL-delimited and verbatim, so a path with a space in it arrives as
    # one entry and a non-ASCII one is not octal-escaped into a name no file has.
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *globs],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = [path for path in out.split("\0") if path]
    assert files, f"the gate's globs matched no files under {paths}"
    return _violations(_run("--json", *files, cwd=REPO_ROOT))


def _lint_paths() -> list[str]:
    """The linted trees as the workflow spells them."""
    match = re.search(r"^\s+LINT_PATHS:\s*(.+)$", WORKFLOW.read_text(), re.MULTILINE)
    assert match, "no LINT_PATHS in the workflow"
    return match.group(1).split()


def _diff_gate_paths() -> list[str]:
    """The pathspec the changed-lines gate passes to `git diff`."""
    match = re.search(r"^paths=\((.+)\)$", DIFF_GATE.read_text(), re.MULTILINE)
    assert match, f"no paths=() assignment in {DIFF_GATE.name}"
    return match.group(1).split()


def _ruff_hook_paths() -> list[str]:
    """The trees ruff-check lints as a floor rather than a scope.

    The comment on the hook in .pre-commit-config.yaml says what that means.
    The lists are still worth comparing; they are just not the same claim.
    """
    block = re.search(
        r"^(\s+)- id: ruff-check$(.*?)(?=^\1- id: |\Z)",
        PRE_COMMIT.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert block, f"no ruff-check hook in {PRE_COMMIT.name}"
    args = re.search(r"^\s+args: \[(.+)\]$", block.group(2), re.MULTILINE)
    assert args, f"the ruff-check hook in {PRE_COMMIT.name} has no args"
    return [
        arg
        for arg in re.findall(r'"([^"]+)"', args.group(1))
        if not arg.startswith("-")
    ]


def _assert_trees_exist(paths: list[str], where: str) -> None:
    """A list of linted trees that names something else has stopped being one.

    This is what catches a flag whose value got separated from it -- rewrite
    `--output-format=full` as two tokens and `full` reads as a tree.
    """
    missing = [path for path in paths if not (REPO_ROOT / path).is_dir()]
    assert not missing, f"{where} lints {missing}, which are not directories here"


@pytest.mark.parametrize(
    "rule", [pytest.param(rule, id=rule) for rule in sorted(FIXTURES)]
)
def test_rule_fires_on_its_fixture(rule: str, tmp_path: Path) -> None:
    """Each rule flags the file written to provoke it.

    The fixture is checked inside a throwaway project carrying this repo's real
    pyproject.toml, so the assertion covers the rule's configuration here and
    not just its implementation upstream.
    """
    relative_path, source = FIXTURES[rule]
    shutil.copy(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    for path, contents in SUPPORT.items():
        support = tmp_path / path
        support.parent.mkdir(parents=True, exist_ok=True)
        support.write_text(contents)
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)

    reported = _violations(_run("--json", str(target), cwd=tmp_path))

    assert rule in reported, (
        f"{rule} did not fire on {relative_path}. Either the fixture no longer "
        f"provokes it or the rule is switched off here. Reported: {sorted(reported)}"
    )


def test_every_rule_has_a_fixture(known_rules: frozenset[str]) -> None:
    """No rule enters or leaves the checker unnoticed."""
    assert set(FIXTURES) == set(known_rules), (
        "fixtures are out of step with xorq-check-style --list; "
        f"missing {sorted(known_rules - set(FIXTURES))}, "
        f"stale {sorted(set(FIXTURES) - known_rules)}"
    )


def test_lint_paths_agree() -> None:
    """One list of linted trees, written in three files that cannot read each other.

    CI reads LINT_PATHS, the changed-lines gate carries its own pathspec so the
    pre-commit hook gets the same one, and ruff-check in .pre-commit-config.yaml
    takes its trees as arguments. Nothing makes one of them read another, so the
    comparison happens here instead. Order is compared too: they are written in
    the same order today, and keeping it that way makes a diff between them
    readable.
    """
    workflow = _lint_paths()
    _assert_trees_exist(workflow, f"LINT_PATHS in {WORKFLOW.name}")
    _assert_trees_exist(_diff_gate_paths(), DIFF_GATE.name)
    _assert_trees_exist(_ruff_hook_paths(), f"ruff-check in {PRE_COMMIT.name}")
    assert _diff_gate_paths() == workflow, (
        f"{DIFF_GATE.name} lints {_diff_gate_paths()}, {WORKFLOW.name} lints {workflow}"
    )
    assert _ruff_hook_paths() == workflow, (
        f"ruff-check in {PRE_COMMIT.name} lints {_ruff_hook_paths()}, "
        f"{WORKFLOW.name} lints {workflow}"
    )


def test_ratchets_name_real_rules(
    known_rules: frozenset[str],
    whole_repo_ratchet: frozenset[str],
    changed_lines_ratchet: frozenset[str],
) -> None:
    """A typo in a --disable list silently enforces a rule nobody chose."""
    assert whole_repo_ratchet <= known_rules, sorted(whole_repo_ratchet - known_rules)
    assert changed_lines_ratchet <= known_rules, sorted(
        changed_lines_ratchet - known_rules
    )


def test_changed_lines_gate_is_the_stricter_one(
    whole_repo_ratchet: frozenset[str], changed_lines_ratchet: frozenset[str]
) -> None:
    """New lines are held to at least the standard the whole tree is."""
    assert changed_lines_ratchet <= whole_repo_ratchet, sorted(
        changed_lines_ratchet - whole_repo_ratchet
    )


def test_enforced_rules_are_clean(
    known_rules: frozenset[str],
    whole_repo_ratchet: frozenset[str],
    repo_violations: frozenset[str],
) -> None:
    """What the whole-repo gate enforces is what CI would pass today."""
    enforced = known_rules - whole_repo_ratchet
    assert enforced, (
        "the ratchet disables every rule; the whole-repo gate enforces nothing"
    )
    assert not (enforced & repo_violations), sorted(enforced & repo_violations)


def test_ratcheted_rules_still_have_violations(
    whole_repo_ratchet: frozenset[str], repo_violations: frozenset[str]
) -> None:
    """The ratchet only turns one way.

    A disabled rule with nothing left to report has been paid off: take it off
    the --disable list in the workflow and it can never come back.
    """
    paid_off = sorted(whole_repo_ratchet - repo_violations)
    assert not paid_off, (
        f"these rules are at zero and should come off the --disable list in "
        f"{WORKFLOW.name}: {paid_off}"
    )


# The gate calls the checker as `uv run --no-sync xorq-check-style`, and the
# throwaway repo below is not a uv project. The checker is already on PATH for
# this suite, so the shim drops the wrapper and execs the rest: what is under
# test is the script's git handling, not uv's resolution.
UV_SHIM = """#!/usr/bin/env bash
[ "$1" = run ] || { echo "unexpected uv invocation: $*" >&2; exit 2; }
shift
[ "$1" = --no-sync ] && shift
exec "$@"
"""


@pytest.fixture
def staged_violation(
    tmp_path: Path, changed_lines_ratchet: frozenset[str], monkeypatch
) -> tuple[Path, str]:
    """A throwaway repo with one staged file the changed-lines gate must flag.

    Returns the rule alongside the repo so the caller can assert the gate
    reported that rule rather than merely exiting non-zero.
    """
    rule = next(r for r in sorted(FIXTURES) if r not in changed_lines_ratchet)
    relative_path, source = FIXTURES[rule]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (shim := bin_dir / "uv").write_text(UV_SHIM)
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(REPO_ROOT / "pyproject.toml", repo / "pyproject.toml")
    # Same support modules as test_rule_fires_on_its_fixture: unstaged, so the
    # gate never checks them, but on disk for the rules that resolve against
    # them. Which rule is picked above moves with the ratchet.
    for path, contents in SUPPORT.items():
        support = repo / path
        support.parent.mkdir(parents=True, exist_ok=True)
        support.write_text(contents)
    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    for args in (["init", "-q"], ["add", "--", relative_path]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    return repo, rule


def _run_diff_gate(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(DIFF_GATE), "--cached"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_diff_gate_fails_on_a_staged_violation(
    staged_violation: tuple[Path, str],
) -> None:
    """The gate both CI and the pre-commit hook run can actually fail.

    Everything else here reads the script's `disable=` and `paths=()` lines as
    text; nothing executes it, and a gate that never fails is the one failure
    mode neither of its two callers would report. The assertion is on the rule
    the checker names, not on the exit code: a broken shim, an unresolved
    `xorq-check-style`, and a `git diff` erroring under `pipefail` all exit
    non-zero too.
    """
    repo, rule = staged_violation
    proc = _run_diff_gate(repo)
    reported = proc.stdout + proc.stderr
    assert f"[{rule}]" in reported, reported
    assert proc.returncode != 0, reported


@pytest.mark.parametrize(
    "marker", ["MERGE_HEAD", "CHERRY_PICK_HEAD", "rebase-merge", "rebase-apply"]
)
def test_diff_gate_stands_down_mid_replay(
    staged_violation: tuple[Path, str], marker: str
) -> None:
    """A replay marker stands the gate down on the same staged violation.

    The stand-down is the branch that switches the whole hook off, so a marker
    the script misspells -- or a `git rev-parse --git-dir` that resolves
    somewhere else, as in a linked worktree -- leaves it on mid-rebase instead.
    """
    repo, _ = staged_violation
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (repo / git_dir / marker).touch()

    proc = _run_diff_gate(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
