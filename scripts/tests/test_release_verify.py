"""Tests for `scripts/release_verify.py`, the guard in front of PyPI.

Each test builds a throwaway repository with a release commit on `main`, so
nothing depends on this repository's tags. TestPyPI is replaced by a dict.

    uv run --no-sync pytest scripts/tests/test_release_verify.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import release_verify  # noqa: E402  (path set above)


WHEEL = "xorq-1.2.3-py3-none-any.whl"
SDIST = "xorq-1.2.3.tar.gz"


def git(cwd: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


def commit(cwd: Path, version: str, subject: str) -> None:
    (cwd / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    git(cwd, "add", "pyproject.toml")
    git(cwd, "commit", "-q", "--allow-empty", "-m", subject)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "release@example.com")
    git(tmp_path, "config", "user.name", "release")
    commit(tmp_path, "1.2.2", "feat: something (#1)")
    commit(tmp_path, "1.2.3", "release: 1.2.3 (#2)")
    git(tmp_path, "tag", "v1.2.3")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / WHEEL).write_bytes(b"wheel")
    (dist / SDIST).write_bytes(b"sdist")
    (dist / ".gitignore").write_text("*\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def uploaded(**overrides: bytes) -> dict[str, str]:
    files = {WHEEL: b"wheel", SDIST: b"sdist", **overrides}
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def run(
    tag: str = "v1.2.3",
    index: Callable[[], dict[str, str] | None] = uploaded,
) -> list[str]:
    return release_verify.check(tag, Path("dist"), "main", lambda _: index())


def test_tested_release_commit_passes(repo: Path) -> None:
    assert run() == []


def test_rebuilt_file_that_differs_from_testpypi_fails(repo: Path) -> None:
    (error,) = run(index=lambda: uploaded(**{SDIST: b"other tree"}))
    assert error.startswith(f"{SDIST}: built ")


def test_file_testpypi_lacks_fails(repo: Path) -> None:
    (error,) = run(index=lambda: {WHEEL: uploaded()[WHEEL]})
    assert error.endswith("TestPyPI missing")


def test_version_never_uploaded_to_testpypi_fails(repo: Path) -> None:
    assert run(index=lambda: None) == [
        "TestPyPI has no xorq 1.2.3; run ci-pre-release first"
    ]


def test_tag_not_on_main_fails(repo: Path) -> None:
    git(repo, "switch", "-q", "-c", "side")
    commit(repo, "1.2.4", "release: 1.2.4 (#3)")
    git(repo, "tag", "v1.2.4")
    errors = run(tag="v1.2.4")
    assert any("is not on main" in error for error in errors)


def test_commit_that_is_not_the_release_merge_fails(repo: Path) -> None:
    commit(repo, "1.2.3", "fix: after the release (#4)")
    git(repo, "tag", "-f", "v1.2.3")
    (error,) = run()
    assert "not the squash merge of a 'release: 1.2.3' PR" in error


def test_pyproject_version_that_disagrees_with_tag_fails(repo: Path) -> None:
    commit(repo, "1.2.4", "release: 1.2.3 (#5)")
    git(repo, "tag", "-f", "v1.2.3")
    assert run() == ["pyproject.toml says 1.2.4, the tag says 1.2.3"]


def test_tag_on_another_commit_than_the_checkout_fails(repo: Path) -> None:
    git(repo, "tag", "-f", "v1.2.3", "HEAD~1")
    errors = run()
    assert any("but the checkout is" in error for error in errors)


@pytest.mark.parametrize(
    "tag",
    [
        pytest.param("1.2.3", id="no-v"),
        pytest.param("v1.2", id="two-parts"),
        pytest.param("v1.2.3rc1", id="pre-release"),
        pytest.param("release-1.2.3", id="branch-name"),
    ],
)
def test_tag_that_is_not_a_plain_version_fails(repo: Path, tag: str) -> None:
    assert run(tag=tag) == [f"tag {tag!r} is not vMAJOR.MINOR.PATCH"]
