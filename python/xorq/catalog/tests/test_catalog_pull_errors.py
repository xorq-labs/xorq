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
    CatalogPullError,
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


# ---------------------------------------------------------------------------
# Pre-flight consistency checks (review feedback #3 — open-items #4).
#
# When the remote tip is in a state that would trip ``assert_consistency``
# (catalog.yaml deleted by hand, malformed, or has an unexpected shape),
# the resolver currently treats the missing/garbage data as legitimate
# input and either silently destroys data (modify/delete revives empty
# defaults → 3-way merge drops every prior entry) or lets a raw parser
# exception leak with no catalog context.
#
# Desired post-fix: pre-flight ``assert_consistency`` against ours and
# against the remote tip before any merge attempt; surface as
# ``CatalogPullError`` naming the corrupt side.
# ---------------------------------------------------------------------------


def test_pull_remote_deleted_catalog_yaml_does_not_silently_drop_entries(two_clones):
    """If the remote tip has deleted ``catalog.yaml`` (manual rm
    bypassing the catalog API), ``pull()`` must refuse rather than
    treating the deletion as 'theirs removed every entry' and 3-way-
    merging the entries list against empty defaults — which silently
    destroys every entry the deleter side never explicitly removed."""
    cat_a, cat_b = two_clones

    # cat_a manually deletes catalog.yaml — bypasses the catalog API
    # entirely (Catalog.push asserts consistency, so we can't use it
    # to push a corrupt state).  Push the broken commit to origin.
    (Path(cat_a.repo.working_dir) / "catalog.yaml").unlink()
    cat_a.repo.git.add("--all")
    cat_a.repo.git.commit("-m", "external: rm catalog.yaml")
    cat_a.repo.git.push("origin", MAIN_BRANCH)

    # cat_b makes a normal commit so the histories diverge and
    # cat_b.pull() exercises the merge path (rather than fast-forward).
    _add_expr_entry(cat_b, "y", value=2)

    with pytest.raises(CatalogPullError):
        cat_b.pull()

    # Pre-fix: pull silently "succeeds" but cat_b's catalog.yaml is
    # rewritten with the empty-defaults theirs side, which drops the
    # boot entry from the entries list.  Post-fix: pre-flight refuses
    # the merge, cat_b's tree is untouched, and boot survives.
    assert "boot" in cat_b.list(), (
        f"pull silently dropped entries: cat_b.list() == {cat_b.list()}"
    )


def test_pull_remote_malformed_yaml_raises_catalog_error(two_clones):
    """If the remote tip has a malformed ``catalog.yaml`` (manual edit
    bypassing the catalog API), ``pull()`` must surface that as a
    catalog-level error rather than letting ``yaml12.parse_yaml``'s raw
    exception leak out of the resolver with no catalog context."""
    cat_a, cat_b = two_clones

    yaml_path = Path(cat_a.repo.working_dir) / "catalog.yaml"
    yaml_path.write_text("[1, 2,\n")  # unclosed flow sequence — fails to parse
    cat_a.repo.git.add("catalog.yaml")
    cat_a.repo.git.commit("-m", "external: corrupt catalog.yaml")
    cat_a.repo.git.push("origin", MAIN_BRANCH)

    _add_expr_entry(cat_b, "y", value=2)

    with pytest.raises(CatalogPullError):
        cat_b.pull()


def test_pull_remote_unexpected_yaml_shape_raises_catalog_error(two_clones):
    """If the remote tip has a ``catalog.yaml`` that parses but isn't a
    dict-or-list (e.g. a scalar — manual edit / corruption), ``pull()``
    must surface that as a catalog-level error rather than letting the
    ``AttributeError`` from ``raw.get(...)`` leak out of the resolver."""
    cat_a, cat_b = two_clones

    yaml_path = Path(cat_a.repo.working_dir) / "catalog.yaml"
    yaml_path.write_text("42\n")  # parses to an int, not a dict
    cat_a.repo.git.add("catalog.yaml")
    cat_a.repo.git.commit("-m", "external: scalar catalog.yaml")
    cat_a.repo.git.push("origin", MAIN_BRANCH)

    _add_expr_entry(cat_b, "y", value=2)

    with pytest.raises(CatalogPullError):
        cat_b.pull()
