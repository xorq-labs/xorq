"""Error-handling tests for ``Catalog.pull()`` — review feedback for
xorq#1902.

These describe the *desired* post-fix behavior; on current code (where
non-conflict ``GitCommandError`` is swallowed and falls through to
``git commit --no-edit``, ``active_branch`` raises ``TypeError`` on a
detached HEAD, and ``CatalogMergeConflict`` doesn't name the failing
remote) every test fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo as GitRepo

import xorq.api as xo
from xorq.catalog.catalog import (
    Catalog,
    CatalogMergeConflict,
)
from xorq.catalog.constants import MAIN_BRANCH
from xorq.catalog.expr_utils import build_expr_context_zip


def _add_expr_entry(catalog, table_name, value, aliases=()):
    expr = xo.memtable({"x": [value]}, name=table_name)
    with build_expr_context_zip(expr) as zip_path:
        entry = catalog.add(zip_path, sync=False, aliases=aliases)
    return entry.name


@pytest.fixture
def two_clones(tmpdir):
    """Two plain-git catalogs sharing a bare origin, both at a boot commit."""
    workdir = Path(tmpdir)
    origin_path = workdir / "origin.git"
    user_a_path = workdir / "user-a"
    user_b_path = workdir / "user-b"

    GitRepo.init(origin_path, bare=True, initial_branch=MAIN_BRANCH)

    cat_a = Catalog.from_repo_path(user_a_path, init=True, annex=False)
    _add_expr_entry(cat_a, "boot", value=0, aliases=("boot-alias",))
    cat_a.repo.create_remote(name="origin", url=str(origin_path))
    cat_a.repo.git.push("-u", "origin", MAIN_BRANCH)

    cat_b = Catalog.clone_from(url=str(origin_path), repo_path=user_b_path, annex=False)
    return cat_a, cat_b


def test_pull_in_detached_head_raises_clear_error(two_clones):
    """When HEAD is detached, ``pull()`` must raise a clear catalog-level
    error rather than a bare ``TypeError`` from gitpython's
    ``active_branch.name`` (the only path to detached HEAD on a catalog
    is external git mucking around; the catalog API itself never
    detaches)."""
    _, cat_b = two_clones
    cat_b.repo.git.checkout("--detach", "HEAD")

    with pytest.raises(Exception) as excinfo:
        cat_b.pull()

    # Pre-fix: gitpython's internal TypeError("HEAD is a detached
    # symbolic reference ...") leaks out. Post-fix: a catalog-level
    # error explaining the user needs a checked-out branch.
    assert not isinstance(excinfo.value, TypeError), (
        f"detached HEAD surfaced as bare TypeError instead of a "
        f"catalog-level error: {excinfo.value!r}"
    )


def test_pull_remote_without_branch_surfaces_cause(tmpdir):
    """Pulling from a remote whose branch doesn't exist must surface
    that as the failure cause (e.g. mention the missing ref or that
    nothing was merged) rather than swallowing the ``git merge`` error
    and falling through to a misleading ``git commit --no-edit``
    follow-up."""
    workdir = Path(tmpdir)
    origin_path = workdir / "origin.git"
    user_path = workdir / "user"

    GitRepo.init(origin_path, bare=True, initial_branch=MAIN_BRANCH)
    cat = Catalog.from_repo_path(user_path, init=True, annex=False)
    _add_expr_entry(cat, "boot", value=0)
    cat.repo.create_remote(name="origin", url=str(origin_path))
    # NOTE: deliberately not pushing — origin has no branches.

    with pytest.raises(Exception) as excinfo:
        cat.pull()

    msg = str(excinfo.value).lower()
    # The downstream "nothing to commit" from the swallowed-GCE
    # fall-through is the bug we're fixing — it masks the real cause.
    assert "nothing to commit" not in msg, (
        f"pull() masked the real failure with a downstream "
        f"'nothing to commit' from git commit --no-edit: {excinfo.value}"
    )


def test_pull_conflict_names_remote_in_error(two_clones):
    """When ``pull()`` raises ``CatalogMergeConflict``, the failing
    remote name must appear in the error — symmetric with #1899's
    ``CatalogPushError(f\"push to remote {remote_name!r} ...\")``."""
    cat_a, cat_b = two_clones
    _add_expr_entry(cat_a, "x", value=1, aliases=("shared",))
    _add_expr_entry(cat_b, "y", value=2, aliases=("shared",))
    cat_a.push()

    with pytest.raises(CatalogMergeConflict) as excinfo:
        cat_b.pull()

    assert "origin" in str(excinfo.value), str(excinfo.value)
