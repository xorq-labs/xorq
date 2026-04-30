"""Smoke tests for tutorial code samples.

Each `docs/tutorials/<...>/<slug>.snippets.py` must execute without error
against the installed xorq. The scripts are extracted (or, where the tutorial
assumes external services like GitHub, narratively adapted) from the
corresponding `.qmd` — when a test fails the fix is either in the tutorial
code or a version bump on a dependency.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"
SCRIPTS = sorted(TUTORIALS_DIR.rglob("*.snippets.py"))

# Slugs whose snippets need extras beyond the default install. The conditions
# are evaluated lazily so missing optional deps just skip the test.
EXTRA_REQUIREMENTS = {
    "build_a_semantic_catalog": ("boring_semantic_layer", "duckdb"),
    "working_with_the_catalog": ("boring_semantic_layer",),
}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_git_annex() -> bool:
    return shutil.which("git-annex") is not None


def _param(script: Path) -> pytest.param:
    slug = script.stem.replace(".snippets", "")
    marks = [pytest.mark.library, pytest.mark.slow]
    for module in EXTRA_REQUIREMENTS.get(slug, ()):
        if not _has_module(module):
            marks.append(
                pytest.mark.skip(reason=f"requires the '{module}' optional dependency")
            )
    if slug == "working_with_the_catalog":
        # Tutorial fetches a public parquet over the network and exercises
        # git-annex on the bare-repo stand-in.
        if not _has_git_annex():
            marks.append(pytest.mark.skip(reason="requires git-annex on PATH"))
    return pytest.param(script, id=slug, marks=marks)


@pytest.mark.parametrize("script", [_param(s) for s in SCRIPTS])
def test_tutorial_snippet_runs(script: Path, tmp_path: Path) -> None:
    # Run from a fresh tempdir with no pyproject.toml ancestor — this matches
    # what a reader in a scratch directory experiences. Each snippet handles
    # its own project setup (mirroring the `uv init` step in the tutorials),
    # so a passing test means the tutorial works for someone outside the
    # xorq source tree, not just for someone who happens to be inside it.
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{script.relative_to(REPO_ROOT)} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
