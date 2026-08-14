"""The backend-name dispatch axis, guarded against renames.

Normalization dispatches on two axes: op type (see ``test_view_rules.py``) and
backend *name string*.  The name axis cannot be covered by a subclass sweep —
it is an open set of strings — and it has already failed exactly the way the
op-type axis failed in gh-2229: gh-1842 shipped
``SnapshotStrategy.normalize_backend`` still holding the project's previous
backend name ``"let"``, two renames after the fact, silently classifying
xorq's own backend as remote.

So the names live in ``xorq.common.constants`` as one canonical set, and these
tests anchor that set to the live ``xorq.backends`` entry-point group: a rename
that lands without updating the constants fails here instead of quietly
under-keying a cache.
"""

from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.backends import _get_backend_names
from xorq.common.constants import (
    BIGQUERY_BACKEND_NAME,
    DATAFUSION_BACKEND_NAMES,
    DISPATCHED_BACKEND_NAMES,
    DUCKDB_BACKEND_NAME,
    NAME_ONLY_BACKEND_NAMES,
    PANDAS_BACKEND_NAME,
    SQLITE_BACKEND_NAME,
)


@pytest.mark.parametrize("name", sorted(set(DISPATCHED_BACKEND_NAMES)))
def test_dispatched_backend_name_is_registered(name: str) -> None:
    """Every name the DT dispatch chain keys on is a live backend.

    This is the gh-1842 tripwire: a renamed backend leaves a string here that
    matches nothing, and the branch it guards becomes dead code that fails open.
    """
    assert name in _get_backend_names()


@pytest.mark.parametrize("name", sorted(set(NAME_ONLY_BACKEND_NAMES)))
def test_name_only_backend_name_is_registered(name: str) -> None:
    assert name in _get_backend_names()


def test_name_only_names_are_a_subset_of_dispatched() -> None:
    """A backend identified by name alone must also be intercepted by the chain.

    Both sets are derived from the same constants; this pins the containment so
    a future edit to one cannot silently desynchronize them.
    """
    assert set(NAME_ONLY_BACKEND_NAMES) <= set(DISPATCHED_BACKEND_NAMES)


def test_constants_are_derived_not_respelled() -> None:
    """The subsets compose from the same atoms (the REMOTE_SCHEMES pattern).

    Guards against someone re-typing a literal into one tuple only.
    """
    assert set(DATAFUSION_BACKEND_NAMES) <= set(DISPATCHED_BACKEND_NAMES)
    assert set(NAME_ONLY_BACKEND_NAMES) == {
        PANDAS_BACKEND_NAME,
        DUCKDB_BACKEND_NAME,
        *DATAFUSION_BACKEND_NAMES,
    }
    assert set(DISPATCHED_BACKEND_NAMES) == {
        *DATAFUSION_BACKEND_NAMES,
        PANDAS_BACKEND_NAME,
        DUCKDB_BACKEND_NAME,
        SQLITE_BACKEND_NAME,
        BIGQUERY_BACKEND_NAME,
    }


def test_xorq_own_backend_is_classified_by_name() -> None:
    """xorq's default connection must land in the name-only set.

    The concrete regression from gh-1842: after the ``let`` -> ``xorq`` ->
    ``xorq_datafusion`` renames, ``xo.connect()`` fell through to the remote
    branch of ``normalize_backend``.
    """
    assert xo.connect().name in NAME_ONLY_BACKEND_NAMES


def test_no_dispatched_name_is_stale_relative_to_upstream_dasher() -> None:
    """The upstream per-backend dict is keyed on the same names.

    ``_dispatch_databasetable`` falls through to
    ``xorq_dasher.rules.expr.normalize_databasetable`` for the backends it does
    not intercept.  Upstream still carries a ``"xorq"`` key that no longer
    matches any registered backend -- assert that dead key stays dead rather
    than silently becoming live again under a future rename.
    """
    assert "xorq" not in _get_backend_names()
