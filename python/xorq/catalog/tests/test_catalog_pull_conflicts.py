"""Conflict-matrix tests for ``Catalog.pull()`` — xorq-labs/xorq#1886 (Bug B).

Each test mirrors a row of the conflict matrix from the issue.  Tests
describe the *desired* post-fix behavior; on current code (where
``catalog.pull()`` bails on divergent branches before any merge attempt)
every test fails.

Two-clone setup throughout:  ``cat_a`` and ``cat_b`` share a bare
origin and start from the same boot commit.  ``cat_a`` does its action
and pushes; ``cat_b`` does its action locally; ``cat_b.pull()`` is the
operation under test.

Auto-resolve cases (pull succeeds silently after fix):
    01  add entry X        | add entry Y                 -> both present
    02  add entry X        | add entry X (same hash)     -> X present
    03  remove entry X     | remove entry Y              -> neither present
    04  remove entry X     | remove entry X              -> X gone
    05  add entry X        | remove entry Y              -> X added, Y gone
    06  add α->X           | add β->Y (different names)  -> both aliases present
    08  add α->X           | add α->X (identical)        -> α->X
    10  remove α           | remove α                    -> α gone
    11  remove α           | remove β                    -> both gone
    12  remove α           | add β->Y (different names)  -> α gone, β added

Genuine conflict cases (pull raises CatalogMergeConflict):
    09  add α->X           | add α->Y (same name, different target)
    13  remove α (in base) | re-point α->Y

(Case 07 — different alias names pointing at the same entry — is a
hygiene smell, not a merge conflict, and is filed as #1901.)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from git import Repo as GitRepo

import xorq.api as xo
from xorq.catalog.catalog import (
    Catalog,
    CatalogAlias,
    CatalogMergeConflict,
)
from xorq.catalog.constants import MAIN_BRANCH
from xorq.catalog.expr_utils import build_expr_context_zip


def _add_expr_entry(catalog, table_name, value, aliases=()):
    """Add a tiny memtable expression as a catalog entry; return its hash name."""
    expr = xo.memtable({"x": [value]}, name=table_name)
    with build_expr_context_zip(expr) as zip_path:
        entry = catalog.add(zip_path, sync=False, aliases=aliases)
    return entry.name


def _remove_alias(catalog, alias):
    CatalogAlias.from_name(alias, catalog).remove()


def _alias_target_hash(catalog, alias):
    return CatalogAlias.from_name(alias, catalog).catalog_entry.name


# Git commit SHAs include the author/commit timestamp at second
# resolution.  When both clones perform the same operation against the
# same content within the same wall-clock second, their commits hash to
# the same SHA and `cat_b.pull()` becomes a trivial fast-forward no-op
# instead of exercising the divergent-merge path the test is meant to
# describe.  Sleep just over a second to guarantee divergent SHAs even
# when the operations are otherwise identical.
_FORCE_DIVERGENT_TIMESTAMP = 1.05


@pytest.fixture
def two_clones(tmpdir):
    """Two plain-git catalogs sharing a bare origin, both at the boot commit.

    Initial state on origin/main:
        entries: ['boot']
        aliases: ['boot-alias' -> boot]
    """
    workdir = Path(tmpdir)
    origin_path = workdir / "origin.git"
    user_a_path = workdir / "user-a"
    user_b_path = workdir / "user-b"

    GitRepo.init(origin_path, bare=True, initial_branch=MAIN_BRANCH)

    cat_a = Catalog.from_repo_path(user_a_path, init=True, annex=False)
    _add_expr_entry(cat_a, "boot", value=0, aliases=("boot-alias",))
    cat_a.repo.create_remote(name="origin", url=str(origin_path))
    cat_a.repo.git.push("-u", "origin", MAIN_BRANCH)

    cat_b = Catalog.clone_from(
        url=str(origin_path), repo_path=user_b_path, annex=False
    )
    return cat_a, cat_b


# ---------------------------------------------------------------------------
# Auto-resolve cases — catalog.pull() should succeed silently after fix
# ---------------------------------------------------------------------------


def test_case_01_add_different_entries(two_clones):
    """Both sides add a different entry hash. The yaml entries: list
    grows in both directions from the common ancestor — stock git's
    line-based diff sees overlapping trailing-region edits and produces
    conflict markers. The yaml-aware resolver must union both adds."""
    cat_a, cat_b = two_clones
    name_x = _add_expr_entry(cat_a, "x", value=1)
    name_y = _add_expr_entry(cat_b, "y", value=2)
    cat_a.push()

    cat_b.pull()

    assert name_x in cat_b.list()
    assert name_y in cat_b.list()
    cat_b.assert_consistency()


def test_case_02_add_same_entry(two_clones):
    """Both sides add the identical entry hash. Identical line additions
    auto-merge in stock git; resolver does not need to fire."""
    cat_a, cat_b = two_clones
    name_x_a = _add_expr_entry(cat_a, "x", value=1)
    time.sleep(_FORCE_DIVERGENT_TIMESTAMP)
    name_x_b = _add_expr_entry(cat_b, "x", value=1)
    assert name_x_a == name_x_b, "test setup: same content must hash the same"
    cat_a.push()

    cat_b.pull()

    assert name_x_a in cat_b.list()
    cat_b.assert_consistency()


def test_case_03_remove_different_entries(two_clones):
    """Both sides remove a different entry that was in the base.
    Resolver propagates both removals via 3-way merge."""
    cat_a, cat_b = two_clones
    name_x = _add_expr_entry(cat_a, "x", value=1)
    name_y = _add_expr_entry(cat_a, "y", value=2)
    cat_a.push()
    cat_b.pull()

    cat_a.remove(name_x, sync=False)
    cat_b.remove(name_y, sync=False)
    cat_a.push()

    cat_b.pull()

    assert name_x not in cat_b.list()
    assert name_y not in cat_b.list()
    cat_b.assert_consistency()


def test_case_04_remove_same_entry(two_clones):
    """Both sides remove the same entry. Idempotent."""
    cat_a, cat_b = two_clones
    name_x = _add_expr_entry(cat_a, "x", value=1)
    cat_a.push()
    cat_b.pull()

    cat_a.remove(name_x, sync=False)
    time.sleep(_FORCE_DIVERGENT_TIMESTAMP)
    cat_b.remove(name_x, sync=False)
    cat_a.push()

    cat_b.pull()

    assert name_x not in cat_b.list()
    cat_b.assert_consistency()


def test_case_05_add_and_remove_unrelated(two_clones):
    """Ours adds X; theirs removes Y (Y was in base; X is fresh).
    Independent edits in different yaml regions."""
    cat_a, cat_b = two_clones
    name_y = _add_expr_entry(cat_a, "y", value=2)
    cat_a.push()
    cat_b.pull()

    name_x = _add_expr_entry(cat_a, "x", value=1)
    cat_b.remove(name_y, sync=False)
    cat_a.push()

    cat_b.pull()

    assert name_x in cat_b.list()
    assert name_y not in cat_b.list()
    cat_b.assert_consistency()


def test_case_06_add_aliases_different_names(two_clones):
    """Ours adds α->X; theirs adds β->Y. Different alias names land at
    distinct symlink paths (no filesystem conflict); only the yaml
    aliases: list overlaps and needs the resolver."""
    cat_a, cat_b = two_clones
    name_x = _add_expr_entry(cat_a, "x", value=1, aliases=("alpha",))
    name_y = _add_expr_entry(cat_b, "y", value=2, aliases=("beta",))
    cat_a.push()

    cat_b.pull()

    assert "alpha" in cat_b.list_aliases()
    assert "beta" in cat_b.list_aliases()
    assert _alias_target_hash(cat_b, "alpha") == name_x
    assert _alias_target_hash(cat_b, "beta") == name_y
    cat_b.assert_consistency()


def test_case_08_add_same_alias_same_target(two_clones):
    """Both sides add the identical alias name pointing at the identical
    entry. Identical adds in both yaml and the symlink path; nothing
    diverges."""
    cat_a, cat_b = two_clones
    name_x_a = _add_expr_entry(cat_a, "x", value=1, aliases=("shared",))
    time.sleep(_FORCE_DIVERGENT_TIMESTAMP)
    name_x_b = _add_expr_entry(cat_b, "x", value=1, aliases=("shared",))
    assert name_x_a == name_x_b
    cat_a.push()

    cat_b.pull()

    assert "shared" in cat_b.list_aliases()
    assert _alias_target_hash(cat_b, "shared") == name_x_a
    cat_b.assert_consistency()


def test_case_10_remove_same_alias(two_clones):
    """Both sides remove the same alias. Idempotent."""
    cat_a, cat_b = two_clones
    _add_expr_entry(cat_a, "x", value=1, aliases=("shared",))
    cat_a.push()
    cat_b.pull()

    _remove_alias(cat_a, "shared")
    time.sleep(_FORCE_DIVERGENT_TIMESTAMP)
    _remove_alias(cat_b, "shared")
    cat_a.push()

    cat_b.pull()

    assert "shared" not in cat_b.list_aliases()
    cat_b.assert_consistency()


def test_case_11_remove_different_aliases(two_clones):
    """Ours removes α; theirs removes β. Both were in the base."""
    cat_a, cat_b = two_clones
    _add_expr_entry(cat_a, "x", value=1, aliases=("alpha",))
    _add_expr_entry(cat_a, "y", value=2, aliases=("beta",))
    cat_a.push()
    cat_b.pull()

    _remove_alias(cat_a, "alpha")
    _remove_alias(cat_b, "beta")
    cat_a.push()

    cat_b.pull()

    assert "alpha" not in cat_b.list_aliases()
    assert "beta" not in cat_b.list_aliases()
    cat_b.assert_consistency()


def test_case_12_remove_alias_and_add_unrelated(two_clones):
    """Ours removes α (in base); theirs adds β->Y (fresh)."""
    cat_a, cat_b = two_clones
    _add_expr_entry(cat_a, "x", value=1, aliases=("alpha",))
    cat_a.push()
    cat_b.pull()

    _remove_alias(cat_a, "alpha")
    name_y = _add_expr_entry(cat_b, "y", value=2, aliases=("beta",))
    cat_a.push()

    cat_b.pull()

    assert "alpha" not in cat_b.list_aliases()
    assert "beta" in cat_b.list_aliases()
    assert _alias_target_hash(cat_b, "beta") == name_y
    cat_b.assert_consistency()


# ---------------------------------------------------------------------------
# Genuine conflict cases — catalog.pull() should raise CatalogMergeConflict
# ---------------------------------------------------------------------------


def test_case_09_add_same_alias_different_targets(two_clones):
    """Both sides add the same alias name pointing at different entries.
    The yaml aliases: list dedupes trivially, but the symlink at
    .xorq/aliases/<name>.zip is an add/add conflict on the same path
    with diverging targets — the only genuine merge conflict we can't
    auto-resolve."""
    cat_a, cat_b = two_clones
    _add_expr_entry(cat_a, "x", value=1, aliases=("shared",))
    _add_expr_entry(cat_b, "y", value=2, aliases=("shared",))
    cat_a.push()

    with pytest.raises(CatalogMergeConflict) as excinfo:
        cat_b.pull()

    conflicted = excinfo.value.conflicted
    assert any("aliases" in p and "shared" in p for p in conflicted), conflicted


def test_case_13_remove_alias_vs_repoint(two_clones):
    """Alias α exists in base. Ours removes α; theirs re-points α to a
    different entry. The symlink at .xorq/aliases/α.zip is a
    modify/delete conflict that survives any auto-resolve."""
    cat_a, cat_b = two_clones
    _add_expr_entry(cat_a, "x", value=1, aliases=("alpha",))
    name_y = _add_expr_entry(cat_a, "y", value=2)
    cat_a.push()
    cat_b.pull()

    _remove_alias(cat_b, "alpha")
    cat_a.add_alias(name_y, "alpha", sync=False)
    cat_a.push()

    with pytest.raises(CatalogMergeConflict) as excinfo:
        cat_b.pull()

    conflicted = excinfo.value.conflicted
    assert any("aliases" in p and "alpha" in p for p in conflicted), conflicted
