"""Offline dialect tests for the Redshift backend.

Sited here rather than under ``python/xorq/backends/redshift/tests/`` for the
reason spelled out in ``test_redshift_backend.py``: ``backends/conftest.py``
auto-applies ``pytest.mark.<backend>`` by path, and every CI job selects by
marker, so a test placed under the backend directory would run only in the
credential-gated workflow. Everything here is ``to_sql`` on an unbound table
and needs no warehouse.

Each test below corresponds to an expression form the rc16 transcript observed
Redshift *rejecting at execution*, after a successful build. The failure mode
these guard against is therefore not "xorq raises" -- it is "xorq emits
confident SQL that the warehouse refuses", which is only visible in the
generated string.

One thing these tests deliberately do NOT claim: that the emitted SQL *runs* on
Redshift. That needs a live warehouse. What is asserted here is the narrower,
fully checkable property -- that the specific construct Redshift is documented
and observed to reject is no longer emitted, and that anything we cannot lower
raises instead of compiling to something structurally invalid.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.backends.redshift.compiler import compiler as redshift_compiler


def to_sql(expr):
    """Compile under the Redshift compiler, with no connection involved."""
    return redshift_compiler.to_sqlglot(expr).sql(
        dialect=redshift_compiler.dialect, pretty=False
    )


@pytest.fixture
def t():
    return xo.table(
        {
            "id": "int64",
            "y": "int64",
            "m": "int64",
            "d": "int64",
            "amt": "float64",
            "grp": "string",
            "flag": "boolean",
        },
        name="t",
    )


def test_date_from_ymd_does_not_emit_make_date(t):
    """:4549 -- ``function make_date(integer,integer,integer) does not exist``.

    ``make_date`` reaches the SQL through the *dialect* layer, not the compiler:
    ``dialects.py`` does ``Postgres.Generator.TRANSFORMS |= {sge.DateFromParts:
    rename_func("make_date")}``. Redshift has no such function.
    """
    sql = to_sql(t.mutate(dt=xo.date(t.y, t.m, t.d)))
    assert "MAKE_DATE" not in sql.upper()
    assert "TO_DATE" in sql.upper()


def test_sum_where_does_not_emit_filter_clause(t):
    """:6725 -- ``FILTER(WHERE ...)`` unsupported.

    ``where=`` is the idiom xorq's own skills teach (:3977), so this is a
    documented construction that could not run. ``AggGen`` already knows how to
    lower it: with ``supports_filter=False`` it rewrites to ``CASE WHEN``
    (``compilers/base.py:139``).
    """
    sql = to_sql(t.group_by("grp").agg(s=t.amt.sum(where=t.flag)))
    assert "FILTER(WHERE" not in sql.upper().replace(" ", "")
    assert "CASE" in sql.upper()


def test_nunique_where_does_not_emit_filter_clause(t):
    """:6725, the ``COUNT(DISTINCT ...)`` spelling of the same defect.

    Worth its own test: the count path builds its aggregate differently from
    ``sum``, so a fix that only covered plain aggregates would leave this one
    emitting ``FILTER`` while the ``sum`` test passed.

    The ``DISTINCT CASE`` assertion is not decoration. The first version of this
    fix emitted ``COUNT(CASE WHEN flag THEN DISTINCT id ELSE NULL END)`` -- no
    ``FILTER``, a ``CASE`` present, and both of the obvious assertions green on
    SQL that is invalid on every engine, because ``DISTINCT`` cannot appear
    inside a ``CASE`` branch. Asserting the *ordering* of the two keywords is
    what distinguishes the fix from that near-miss.
    """
    sql = to_sql(t.group_by("grp").agg(n=t.id.nunique(where=t.flag)))
    upper = sql.upper()
    assert "FILTER(WHERE" not in upper.replace(" ", "")
    assert "CASE" in upper
    assert "COUNT(DISTINCT CASE" in upper.replace("  ", " ")
    assert "THEN DISTINCT" not in upper


def test_ranking_window_emits_no_frame_clause(t):
    """:5440 -- ``Frame clause should not be specified for ranking window
    functions``.

    :5475 records that *both* the windowed and unwindowed spellings emitted
    ``ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING``, so no API
    spelling avoided it -- which is why this is a compiler fix rather than a
    documentation note.
    """
    expr = t.mutate(rn=xo.row_number().over(xo.window(group_by=t.grp, order_by=t.id)))
    sql = to_sql(expr)
    assert "ROW_NUMBER()" in sql.upper()
    assert "ROWS BETWEEN" not in sql.upper()
    assert "RANGE BETWEEN" not in sql.upper()


def test_non_ranking_window_keeps_its_frame_clause(t):
    """The control for the test above.

    Suppressing the frame clause for *every* window function would silently
    change the semantics of cumulative aggregates, which genuinely need their
    frame. Redshift's restriction is specific to ranking functions, and so is
    the fix.
    """
    expr = t.mutate(
        running=t.amt.sum().over(
            xo.window(group_by=t.grp, order_by=t.id, preceding=None, following=0)
        )
    )
    sql = to_sql(expr)
    assert "SUM(" in sql.upper()
    assert "ROWS BETWEEN" in sql.upper()


def test_distinct_on_raises_rather_than_emitting_first(t):
    """:5507 -- ``.distinct(on=[...])`` compiled to ``FIRST()``, which Redshift
    has no aggregate for.

    Redshift *does* have ``FIRST_VALUE``, but only as a window function, and
    this lowering puts the call in a ``GROUP BY`` aggregate position where a
    window function is not allowed. There is no in-place substitution, so the
    honest behaviour is to raise at compile time -- the whole point of the
    issue is that this currently fails at *execution*, after the models are
    written, documented and built.
    """
    with pytest.raises(com.UnsupportedOperationError, match="(?i)first"):
        to_sql(t.distinct(on=["grp"], keep="first"))


def test_transforms_do_not_depend_on_import_order():
    """The hazard the ``RedshiftCompiler`` docstring named, pinned as a test.

    Verified 2026-09-23: sqlglot's ``Redshift.Generator`` copies
    ``Postgres.Generator.TRANSFORMS`` at class-creation time, while
    ``xorq.vendor.ibis.backends.sql.dialects`` mutates that same dict in place
    afterwards. Importing sqlglot's Redshift *before* xorq's dialects yielded
    190 transforms; importing it *after* yielded 195. Same process, same
    versions, different dialect behaviour -- decided purely by which module
    loaded first.

    xorq's dialect is order-independent because ``dialects.py`` imports
    sqlglot's Redshift at module top, forcing that class to be built before the
    mutation runs, and then copies the dict explicitly. This test runs the two
    orders in separate subprocesses and asserts they agree; it fails if anyone
    reintroduces a dialect that inherits TRANSFORMS lazily.

    It must import the *compiler*, not ``xorq.api``: importing ``xorq.api``
    alone does not trigger the in-place mutation, so a version of this test
    written against it would get the same answer in both arms and pass
    vacuously.
    """
    probe = textwrap.dedent(
        """
        import sys
        order = sys.argv[1]
        if order == "sqlglot-first":
            import sqlglot.dialects.redshift  # noqa: F401
            import xorq.vendor.ibis.backends.sql.dialects  # noqa: F401
        else:
            import xorq.vendor.ibis.backends.sql.dialects  # noqa: F401
            import sqlglot.dialects.redshift  # noqa: F401
        from xorq.backends.redshift.compiler import compiler
        names = sorted(k.__name__ for k in compiler.dialect.Generator.TRANSFORMS)
        print(len(names))
        print(",".join(names))
        """
    )

    def transforms_under(order):
        out = subprocess.run(
            [sys.executable, "-c", probe, order],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()

    assert transforms_under("sqlglot-first") == transforms_under("xorq-first")
