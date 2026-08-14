"""The backend-name dispatch axis, guarded against renames.

Normalization dispatches on two axes: op type (see ``test_view_rules.py``) and
backend *name string*.  The name axis cannot be covered by a subclass sweep —
it is an open set of strings — and it has already failed exactly the way the
op-type axis failed in gh-2229: gh-1842 shipped
``SnapshotStrategy.normalize_backend`` still holding the project's previous
backend name ``"let"``, two renames after the fact, silently classifying
xorq's own backend as remote.

So the names live in the ``BackendName`` enum (``xorq.common.enums``) with the
dispatch sets in ``xorq.common.constants`` derived from its members, and these
tests anchor the enum to the live ``xorq.backends`` entry-point group: a rename
that lands without updating the enum fails here instead of quietly under-keying
a cache.
"""

from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.backends import _get_backend_names
from xorq.common.constants import (
    DATAFUSION_BACKEND_NAMES,
    DISPATCHED_BACKEND_NAMES,
    NAME_ONLY_BACKEND_NAMES,
)
from xorq.common.enums import BackendName


@pytest.mark.parametrize("name", sorted(BackendName))
def test_backend_name_is_registered(name: BackendName) -> None:
    """Every ``BackendName`` member is a live backend.

    This is the gh-1842 tripwire: a renamed backend leaves a member here that
    matches nothing, and every branch it guards becomes dead code that fails
    open.  Sweeping the enum itself means a newly added member is covered
    automatically, with no aggregate tuple to forget to update.
    """
    assert name in _get_backend_names()


def test_dispatch_sets_are_derived_from_the_enum() -> None:
    """The dispatch sets compose from ``BackendName`` members, never literals.

    Guards against someone re-typing a raw string into one set only — the
    respelling that let gh-1842's stale name survive two renames.
    """
    assert DISPATCHED_BACKEND_NAMES == frozenset(BackendName)
    assert NAME_ONLY_BACKEND_NAMES <= DISPATCHED_BACKEND_NAMES
    assert set(DATAFUSION_BACKEND_NAMES) <= NAME_ONLY_BACKEND_NAMES
    assert NAME_ONLY_BACKEND_NAMES == {
        BackendName.PANDAS,
        BackendName.DUCKDB,
        *DATAFUSION_BACKEND_NAMES,
    }


def test_backend_names_are_plain_strings() -> None:
    """Members must be drop-in for the ``str`` names backends report.

    ``dt.source.name`` comparisons and ``NAME_ONLY_BACKEND_NAMES`` membership
    tests run against plain strings, and anything folded into a hash token must
    stay a plain string — pin the value semantics ``StrEnum`` provides.
    """
    for member in BackendName:
        assert isinstance(member, str)
        assert str(member) == member.value
        assert hash(member) == hash(member.value)


def test_xorq_own_backend_is_classified_by_name() -> None:
    """xorq's default connection must land in the name-only set.

    The concrete regression from gh-1842: after the ``let`` -> ``xorq`` ->
    ``xorq_datafusion`` renames, ``xo.connect()`` fell through to the remote
    branch of ``normalize_backend``.
    """
    assert xo.connect().name in NAME_ONLY_BACKEND_NAMES


def test_retired_backend_name_stays_retired() -> None:
    """No registered backend is named ``"xorq"`` again.

    ``_dispatch_databasetable`` falls through to
    ``xorq_dasher.rules.expr.normalize_databasetable`` for the backends it does
    not intercept, and upstream's per-backend dict still carries a ``"xorq"``
    key from before the ``xorq_datafusion`` rename.  If a backend named
    ``"xorq"`` ever registers again, that stale upstream normalizer silently
    becomes live for it -- fail here first.
    """
    assert "xorq" not in _get_backend_names()
