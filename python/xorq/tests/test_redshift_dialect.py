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
import sqlglot

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.backends.redshift.compiler import compiler as redshift_compiler
from xorq.vendor.ibis.backends.sql.datatypes import PostgresType, RedshiftType


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


def test_sqlglot_redshift_is_built_before_the_postgres_transforms_mutation():
    """The eager import in ``dialects.py`` is the whole of the protection.

    sqlglot's ``Redshift.Generator`` copies ``Postgres.Generator.TRANSFORMS`` at
    class-creation time; ``dialects.py`` mutates that same dict in place at
    module level. Whichever runs first wins. ``dialects.py`` therefore imports
    sqlglot's Redshift at the top of the file -- *above* the mutation -- which
    forces the copy to be taken from the pre-mutation dict no matter what any
    other module does.

    The predecessor of this test ran the two import orders in subprocesses and
    compared them. It could not fail: both arms imported
    ``xorq.vendor.ibis.backends.sql.dialects``, whose top-level import forces
    sqlglot's class in *both* orders, so the two arms were the same experiment
    run twice. Measured against the unfixed compiler it was green.

    This asserts the protective property directly instead. Remove the eager
    import from ``dialects.py`` and the first assertion fails; move the
    mutation above it and the second does. Counterfactual confirmed by
    mutating ``Postgres.Generator.TRANSFORMS`` before importing
    ``sqlglot.dialects.redshift`` in a scratch process: all five names leak.
    """
    probe = textwrap.dedent(
        """
        import sys

        import sqlglot.expressions as sge

        import xorq.vendor.ibis.backends.sql.dialects  # noqa: F401

        assert "sqlglot.dialects.redshift" in sys.modules, "eager-import-gone"

        from sqlglot.dialects.redshift import Redshift as SqlglotRedshift

        postgres_only = (
            sge.Split,
            sge.RegexpSplit,
            sge.DateFromParts,
            sge.ArraySize,
            sge.Pow,
        )
        print(
            ",".join(
                sorted(
                    k.__name__
                    for k in postgres_only
                    if k in SqlglotRedshift.Generator.TRANSFORMS
                )
            )
        )
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", (
        "postgres-only renames leaked into sqlglot's own Redshift generator: "
        f"{out.stdout.strip()}"
    )


def test_count_where_does_not_put_star_inside_a_case(t):
    """``COUNT(CASE WHEN c THEN * ELSE NULL END)`` is not valid anywhere.

    ``supports_filter=False`` makes ``AggGen`` wrap every argument in a
    ``CASE``, and ``visit_CountStar``'s argument is ``STAR``. This is the most
    frequently reached ``where=`` path in the codebase, and unlike the
    ``FILTER`` spelling it replaced, it does not even parse.
    """
    sql = to_sql(t.group_by("grp").agg(n=t.count(where=t.flag)))
    assert "THEN *" not in sql
    assert "FILTER(" not in sql
    assert "COUNT(CASE WHEN" in sql
    sqlglot.parse_one(sql, dialect="redshift")


def test_count_star_without_a_predicate_stays_count_star(t):
    """The fix above must not cost the unfiltered spelling its ``COUNT(*)``."""
    assert "COUNT(*)" in to_sql(t.group_by("grp").agg(n=t.count()))


def test_table_nunique_where_keeps_distinct_outside_the_case(t):
    """``Table.nunique(where=...)`` reaches ``visit_CountDistinctStar``.

    That is a *different* method from ``visit_CountDistinct``, inherited
    unchanged from the postgres compiler, and it hit the identical
    ``DISTINCT``-inside-``CASE`` trap the column-level override exists to
    prevent.
    """
    sql = to_sql(t.aggregate(n=t.nunique(where=t.flag)))
    assert "COUNT(DISTINCT CASE WHEN" in sql
    assert "THEN DISTINCT" not in sql
    sqlglot.parse_one(sql, dialect="redshift")


def test_arbitrary_emits_any_value_not_first(t):
    """``ops.Arbitrary: "first"`` is inherited from the postgres ``SIMPLE_OPS``.

    ``__init_subclass__`` generates ``visit_Arbitrary`` from that mapping, so
    ``.arbitrary()`` emitted the very ``FIRST()`` that ``visit_First`` raises
    on -- a method raising does not stop a sibling being generated.
    """
    sql = to_sql(t.group_by("grp").agg(a=t.amt.arbitrary()))
    assert "ANY_VALUE(" in sql
    assert "FIRST(" not in sql


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda t: t.amt.median(where=t.flag), id="median"),
        pytest.param(lambda t: t.amt.quantile(0.9, where=t.flag), id="quantile"),
        pytest.param(lambda t: t.amt.approx_median(where=t.flag), id="approx_median"),
        pytest.param(lambda t: t.grp.mode(where=t.flag), id="mode"),
    ],
)
def test_ordered_set_aggregates_emit_no_filter_clause(t, build):
    """Ordered-set aggregates build ``sge.Filter`` by hand and never see AggGen.

    So ``supports_filter=False`` does not reach them: these four kept emitting
    byte-identical PostgreSQL after the flag flip. The predicate belongs inside
    the ``WITHIN GROUP`` ordering instead, where NULL-skipping makes it
    equivalent.
    """
    sql = to_sql(t.group_by("grp").agg(q=build(t)))
    assert "FILTER(" not in sql
    assert "WITHIN GROUP (ORDER BY CASE WHEN" in sql
    sqlglot.parse_one(sql, dialect="redshift")


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda t: t.amt.argmax(t.id), id="argmax"),
        pytest.param(lambda t: t.amt.argmin(t.id), id="argmin"),
        pytest.param(lambda t: t.amt.argmax(t.id, where=t.flag), id="argmax_where"),
    ],
)
def test_argminmax_raises_rather_than_emitting_array_agg(t, build):
    """Redshift has no ``ARRAY_AGG``, so the postgres lowering has no target.

    ``visit_ArgMinMax`` also synthesizes a non-``None`` ``where`` for its null
    guards, so with ``supports_filter=False`` *every* call -- predicate or not
    -- produced ``CASE WHEN ... THEN x ORDER BY k DESC ELSE NULL END``, which
    does not parse. Raising names the ``row_number()`` alternative instead.
    """
    with pytest.raises(com.UnsupportedOperationError, match="(?i)array_agg"):
        to_sql(t.group_by("grp").agg(a=build(t)))


def test_group_concat_where_leaves_the_separator_alone(t):
    """Redshift requires ``LISTAGG``'s delimiter to be a constant.

    The generated ``SIMPLE_OPS`` reduction impl passes every argument to
    ``AggGen``, which CASE-wrapped the separator alongside the value.
    """
    sql = to_sql(t.group_by("grp").agg(g=t.grp.group_concat(",", where=t.flag)))
    assert "LISTAGG(CASE WHEN" in sql
    assert "', '" not in sql
    assert sql.count("CASE WHEN") == 1, sql
    sqlglot.parse_one(sql, dialect="redshift")


def test_group_concat_order_by_uses_within_group(t):
    """PostgreSQL spells this ``string_agg(x, sep ORDER BY k)``; Redshift does
    not accept an ``ORDER BY`` inside the argument list."""
    sql = to_sql(t.group_by("grp").agg(g=t.grp.group_concat(",", order_by=t.id)))
    assert "WITHIN GROUP (ORDER BY" in sql
    assert 'LISTAGG("t0"."grp", \',\')' in sql
    sqlglot.parse_one(sql, dialect="redshift")


@pytest.mark.parametrize("method", ["lag", "lead"])
def test_offset_window_functions_emit_no_frame_clause(t, method):
    """AWS documents ``LAG``/``LEAD`` with no frame clause slot.

    Unlike the ranking family this is documentation-derived rather than an
    observed rejection, but it is safe on its own terms: an offset function
    reads a row at a fixed displacement and cannot depend on the frame, so
    removing the clause is observable only in the emitted string.
    """
    w = xo.window(group_by=t.grp, order_by=t.id)
    sql = to_sql(t.mutate(out=getattr(t.amt, method)().over(w)))
    assert "ROWS BETWEEN" not in sql
    assert f"{method.upper()}(" in sql


def test_compiler_type_mapper_is_redshifts(t):
    """``TYPE_MAPPERS[dialect_name]`` and ``compiler.type_mapper`` are two
    separate bindings, and only the first was set.

    The second is what ``Backend.get_schema`` and ``_get_schema_using_query``
    use to parse type strings coming back from the warehouse, so a ``VARBYTE``
    column was read back as ``unknown``.
    """
    assert redshift_compiler.type_mapper is RedshiftType
    assert str(RedshiftType.from_string("varbyte")) == "binary"
    assert str(PostgresType.from_string("varbyte")) == "unknown"
