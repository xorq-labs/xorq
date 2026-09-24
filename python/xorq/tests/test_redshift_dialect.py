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

SETTLED AGAINST A LIVE WAREHOUSE, 2026-09-24
--------------------------------------------

Three claims in this module once rested on AWS's documented grammar rather than
on an observed rejection. All three were settled against the xorq-test
warehouse (Redshift Serverless, us-east-2) on 2026-09-24, and all three came
back confirming the behaviour this module already had. The queries live in
``scripts/2026-09-24-redshift-settle-open-questions.sh`` and its round-2
companion, so the answers are reproducible rather than remembered.

1. **Does Redshift decode ``\t`` in a plain string literal? YES.**
   ``SELECT LENGTH('\t')`` is 1 and ``'\t' = CHR(9)`` is true. So the escape
   rules that came with the dialect are correct, ``.strip()``'s TRIM literal is
   right, and -- the part nobody expected -- the regex change is a REAL BUG FIX
   rather than a risk: ``'1' ~ '\\\\d'`` matches and ``'1' ~ '\\d'`` does not,
   while ``'d' ~ '\\d'`` does. Under the old Postgres dialect this backend was
   emitting ``'\\d'``, i.e. sending the regex "literal letter d" every time a
   user wrote ``\\d``, silently, against a live warehouse.

2. **Do LAG/LEAD reject a frame clause? YES.** "Frame clause should not be
   specified for window function lag". ``_OFFSET_OPS`` is necessary, not
   defensive.

3. **Do LISTAGG/PERCENTILE_CONT/MEDIAN reject ORDER BY and a frame in window
   position? YES.** "window specification should not contain frame clause and
   order-by for window function median". ``_PARTITION_ONLY_OPS`` is necessary.
   ``ANY_VALUE`` is separately "Window function any_value not supported",
   which is why it raises in window position.

The same session settled four things that were NOT open questions, because
nobody had thought to ask -- each is now a raise in the compiler:
``Table.nunique()`` (no multi-column COUNT DISTINCT and no record equality),
``.mode()`` (no MODE in any spelling), ``.quantile()`` over a non-numeric
column (PERCENTILE_DISC unsupported outright), and confirmation that
``ARRAY_AGG``, ``FIRST``, ``STARTS_WITH``, ``MAKE_DATE`` and
``DATE_FROM_PARTS`` are all absent, as the compiler already assumed.

One item remains open and is NOT a dialect question: the two unverified type
widths in ``test_redshift_backend.py`` (unbounded ``VARCHAR``, the
``TIMESTAMP(6)`` precision modifier).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import subprocess
import sys
import textwrap

import pytest
import sqlglot


# Must run BEFORE the xorq.backends.redshift import below. That module reaches
# vendor/ibis/backends/postgres, which imports psycopg unguarded, and psycopg
# ships in the ``postgres`` extra rather than in the core dependencies. CI runs
# ``pytest -m <backend>`` with no path filter, so every job COLLECTS this file,
# and the jobs without that extra failed collection outright rather than
# deselecting -- a red build that says ModuleNotFoundError, not a skip.
# ``backends/conftest.py`` guards ``backends/<name>/`` paths only, and this file
# is deliberately sited outside them (see the docstring above), so the guard has
# to be here. The E402s are that guard running first, not import sloppiness.
#
# TWO names, and which two is measured per-name rather than inferred. Blocking
# each candidate independently against these modules:
#
#     adbc_driver_manager     blocked -> 2 collection errors   REQUIRED
#     psycopg                 blocked -> 2 skipped             REQUIRED
#     adbc_driver_postgresql  blocked -> 59 passed             NOT required
#
# The third name is deliberately absent, and the reason is structural rather
# than situational: it is reached only through
# ``common/utils/postgres_utils.py:1``, a chain these two modules never take.
# The two names above ARE requirements of the chain they do take, which is why
# guarding them costs nothing.
#
# Calibrated honestly, because today the distinction is latent rather than
# visible: psycopg ships only in the ``postgres`` and ``redshift`` extras, both
# of which also ship adbc-driver-postgresql, which in turn depends on
# adbc-driver-manager. So in every configuration that exists right now all three
# names are satisfied together, the guard fires as a unit, and a third name
# would be redundant rather than actively over-skipping. What makes leaving it
# out worth doing is ``pyproject.toml`` inside the redshift extra -- "whether a
# psycopg-only extra should exist is open". If that lands, a three-name guard
# skips this entire file, every dialect and inventory test and none of them
# touching ADBC, in exactly the configuration that extra exists to create.
#
# An earlier version of this guard named psycopg alone, reasoning from one CI
# job's traceback -- that job happened to have adbc_driver_manager installed, so
# the traceback could not show the dependency. A traceback reports the
# environment it ran in, not the requirement.
pytest.importorskip("adbc_driver_manager")
pytest.importorskip("psycopg")

import xorq.api as xo  # noqa: E402
import xorq.common.exceptions as com  # noqa: E402
from xorq.backends.redshift.compiler import RedshiftCompiler  # noqa: E402
from xorq.backends.redshift.compiler import (  # noqa: E402
    compiler as redshift_compiler,
)
from xorq.vendor.ibis.backends.sql.datatypes import (  # noqa: E402
    PostgresType,
    RedshiftType,
)


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
            "s": "string",
            "arr": "array<int64>",
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


def test_table_nunique_raises_in_either_spelling(t):
    """``Table.nunique()`` has no Redshift lowering at all.

    VERIFIED on the xorq-test warehouse 2026-09-24: ``COUNT(DISTINCT a, b)``
    is "function count(bigint, varchar) does not exist" and
    ``COUNT(DISTINCT (a, b))`` is "could not identify an equality operator for
    type record". sqlglot's ``MULTI_ARG_DISTINCT = True`` for Redshift claims
    the first; the warehouse disagrees.

    This test replaces one that asserted the emitted string kept ``DISTINCT``
    outside the ``CASE``. That assertion was true, and the SQL it described
    parsed cleanly, and it could never have run. It is the sharpest example in
    this module of how far a parse oracle gets you: it proved the string was
    well-formed and said nothing about whether Redshift would accept it.
    """
    for expr in (t.aggregate(n=t.nunique()), t.aggregate(n=t.nunique(where=t.flag))):
        with pytest.raises(com.UnsupportedOperationError, match="(?i)nunique"):
            to_sql(expr)


def test_single_column_nunique_still_works(t):
    """The blanket above is scoped to the whole-table form."""
    sql = to_sql(t.group_by("grp").agg(n=t.id.nunique(where=t.flag)))
    assert "COUNT(DISTINCT CASE WHEN" in sql
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


def test_mode_raises(t):
    """Redshift has no ``MODE`` in any spelling.

    VERIFIED on the xorq-test warehouse 2026-09-24: ``MODE() WITHIN GROUP`` is
    a syntax error at "WITHIN", and ``MODE(title)`` is "function mode(varchar)
    does not exist".

    An earlier round fixed this method's ``FILTER (WHERE ...)`` clause and
    deliberately declined to say whether the function existed, on the grounds
    that the narrower fix asserted less. With a warehouse, the answer is that
    the whole method had no target.
    """
    with pytest.raises(com.UnsupportedOperationError, match="(?i)mode"):
        to_sql(t.group_by("grp").agg(m=t.grp.mode(where=t.flag)))


def test_quantile_over_a_non_numeric_column_raises(t):
    """``percentile_disc`` is rejected outright, and ``percentile_cont``
    refuses a non-numeric ordering expression.

    VERIFIED on the xorq-test warehouse 2026-09-24: PERCENTILE_DISC gives
    'Aggregate function "percentile_disc" is not supported; use approximate
    percentile_disc or percentile_cont instead', and PERCENTILE_CONT over a
    varchar gives 'Non supported data-type in order-by expression'. So the
    ``disc`` branch of ``visit_Quantile`` had no valid target either way.
    """
    with pytest.raises(com.UnsupportedOperationError, match="(?i)percentile_disc"):
        to_sql(t.group_by("grp").agg(q=t.grp.quantile(0.5)))


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
    # The separator appears exactly once, as a bare literal. Asserting on a
    # spelling the separator never has (this line previously read
    # `assert "\', \'" not in sql`, with a space, against a separator of ",")
    # is an assertion that cannot fail.
    assert sql.count("','") == 1, sql
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


def test_last_raises_like_first(t):
    """``visit_Last`` had no coverage at all: the ``distinct(on=...)`` test only
    exercised ``keep="first"``, and nothing reached ``.last()``."""
    with pytest.raises(com.UnsupportedOperationError, match="(?i)last"):
        to_sql(t.distinct(on=["grp"], keep="last"))


def test_approx_nunique_where_keeps_distinct_outside_the_case(t):
    """The third entry point into the ``DISTINCT``-inside-``CASE`` trap.

    ``visit_CountDistinct`` and ``visit_CountDistinctStar`` were each fixed on
    discovery; ``visit_ApproxCountDistinct`` (``compilers/postgres.py:273``)
    does the identical thing and was reached by neither. Found by sweeping
    every ``where=``-taking reduction rather than by reasoning from the
    override list -- which is what ``test_every_filtered_reduction_parses``
    below now does on every run.
    """
    sql = to_sql(t.group_by("grp").agg(n=t.id.approx_nunique(where=t.flag)))
    assert "COUNT(DISTINCT CASE WHEN" in sql
    assert "THEN DISTINCT" not in sql
    sqlglot.parse_one(sql, dialect="redshift")


def test_startswith_does_not_build_an_unescaped_like_pattern(t):
    """``STARTS_WITH`` does not exist on Redshift and sqlglot lowers it to a
    ``LIKE`` whose pattern is the operand plus ``'%'``, unescaped.

    A ``%`` or ``_`` in the operand then matches as a wildcard, so this is
    silent wrong rows rather than an error. ``endswith`` is immune because it
    compares extracted text instead of building a pattern; ``startswith`` now
    has the same shape.
    """
    sql = to_sql(t.select(o=t.s.startswith("a%")))
    assert "LIKE" not in sql
    assert "LEFT(" in sql
    sqlglot.parse_one(sql, dialect="redshift")


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda t: t.select(o=t.arr.sort()), id="sort"),
        pytest.param(lambda t: t.select(o=t.arr.unique()), id="unique"),
        pytest.param(lambda t: t.select(o=t.arr.unnest()), id="unnest"),
        pytest.param(lambda t: t.select(o=t.arr.sums()), id="sums"),
        pytest.param(lambda t: t.select(o=t.arr.map(lambda x: x + 1)), id="map"),
        pytest.param(lambda t: t.select(o=t.arr.filter(lambda x: x > 1)), id="filter"),
        pytest.param(lambda t: t.select(o=t.arr.union(t.arr)), id="union"),
        pytest.param(lambda t: t.select(o=t.arr.index(1)), id="index"),
        pytest.param(lambda t: t.group_by("grp").agg(c=t.amt.collect()), id="collect"),
        pytest.param(lambda t: t.select(o=t.s.re_split(",")), id="re_split"),
    ],
)
def test_unnest_dependent_ops_raise_instead_of_emitting_holes(t, build):
    """sqlglot's Redshift generator warns and returns ``""`` for ``UNNEST``.

    That empty string flows back into expression building, so the observed
    failures were an internal ``AttributeError: 'str' object has no attribute
    'args'`` raised from inside sqlglot, or SQL with a hole in it --
    ``SELECT  AS "o"``, ``ARRAY(SELECT DISTINCT)``,
    ``ARRAY(SELECT UNION SELECT)``, an array column in the ``FROM`` position.
    None of these were ever going to run on Redshift, which has no array type;
    what they cost was the diagnosis, since none named the backend or the op.
    """
    with pytest.raises(com.OperationNotDefinedError):
        to_sql(build(t))


@pytest.mark.parametrize(
    ("build", "kept"),
    [
        pytest.param(lambda t: t.arr.length(), "GET_ARRAY_LENGTH", id="length"),
        pytest.param(lambda t: t.arr[0], "[", id="index"),
        pytest.param(lambda t: t.arr + t.arr, "ARRAY_CONCAT", id="concat"),
        pytest.param(lambda t: t.s.split(","), "SPLIT_TO_ARRAY", id="split"),
    ],
)
def test_array_ops_with_a_redshift_spelling_are_kept(t, build, kept):
    """The blanket above must not swallow the ops that do lower.

    ``SPLIT_TO_ARRAY`` and ``GET_ARRAY_LENGTH`` are also the only coverage the
    ``Redshift.Generator.TRANSFORMS`` block in ``dialects.py`` has: before this
    test the whole block could be deleted with all tests still green.
    """
    assert kept in to_sql(t.select(o=build(t)))


def test_pow_needs_no_transform_override(t):
    """``sge.Pow: rename_func("power")`` was dead code -- sqlglot's own Redshift
    generator already renders ``POWER(...)``. Removed; this pins the behaviour
    it was pretending to provide."""
    assert "POWER(" in to_sql(t.select(o=t.amt**2))


@pytest.mark.parametrize(
    ("build", "func"),
    [
        pytest.param(lambda t: t.grp.group_concat(","), "LISTAGG", id="listagg"),
        pytest.param(lambda t: t.amt.median(), "PERCENTILE_CONT", id="median"),
        pytest.param(lambda t: t.amt.quantile(0.9), "PERCENTILE_CONT", id="quantile"),
    ],
)
def test_partition_only_window_functions_drop_order_by_and_frame(t, build, func):
    """Redshift documents these in window position as ``OVER ([PARTITION BY])``.

    No ``ORDER BY``, no frame -- their ordering rides in ``WITHIN GROUP``
    instead. So unlike the ranking family, where only the frame is dropped,
    the whole ``OVER`` body below ``PARTITION BY`` has to go.
    """
    w = xo.window(group_by=t.grp, order_by=t.id)
    sql = to_sql(t.mutate(o=build(t).over(w)))
    assert func in sql
    assert 'OVER (PARTITION BY "t0"."grp")' in sql
    assert "ROWS BETWEEN" not in sql


def test_arbitrary_over_a_window_raises(t):
    """``ANY_VALUE`` is an aggregate on Redshift, not a window function."""
    w = xo.window(group_by=t.grp, order_by=t.id)
    with pytest.raises(com.UnsupportedOperationError, match="(?i)any_value"):
        to_sql(t.mutate(o=t.amt.arbitrary().over(w)))


def test_cumulative_aggregates_keep_their_frame(t):
    """The two suppression lists must stay scoped: a frame dropped here would
    be a silent wrong answer rather than a loud one."""
    w = xo.window(group_by=t.grp, order_by=t.id)
    assert "ROWS BETWEEN" in to_sql(t.mutate(o=t.amt.sum().over(w)))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda t: t.group_by("grp").agg(n=t.id.nunique()), id="nunique_col"
        ),
        pytest.param(lambda t: t.group_by("grp").agg(q=t.amt.median()), id="median"),
        pytest.param(
            lambda t: t.group_by("grp").agg(g=t.grp.group_concat(",")), id="gc"
        ),
    ],
)
def test_the_unfiltered_branch_of_each_override_still_works(t, build):
    """Every override above forks on ``where is None``; only the ``is not None``
    side was exercised."""
    sql = build(t)
    assert "CASE WHEN" not in to_sql(sql)
    sqlglot.parse_one(to_sql(sql), dialect="redshift")


def test_every_filtered_reduction_parses(t):
    """The sweep, as a standing guard rather than a one-off.

    Three separate findings -- ``visit_CountDistinct``,
    ``visit_CountDistinctStar``, ``visit_ApproxCountDistinct`` -- were one
    mechanism reached through three entry points, and the first two were found
    by inspection while the third was found only by compiling everything.
    ``agg = AggGen(supports_filter=False)`` changes the lowering of *every*
    reduction on this backend, so the set that needs checking is every
    reduction, not the set someone thought to override.

    Parseability is a weak oracle -- it cannot tell you Redshift will accept a
    well-formed string. It is the strongest one available offline, and it is
    exactly the property the CASE fallback breaks.
    """
    columns = {"int": t.id, "float": t.amt, "string": t.grp, "bool": t.flag}
    checked = 0
    failures = []
    for dtype_name, col in columns.items():
        for name in sorted(dir(col)):
            if name.startswith("_"):
                continue
            method = getattr(type(col), name, None)
            if not callable(method):
                continue
            try:
                takes_where = "where" in inspect.signature(method).parameters
            except (TypeError, ValueError):
                continue
            if not takes_where:
                continue
            for predicate in (None, t.flag):
                try:
                    expr = t.group_by("grp").agg(
                        out=getattr(col, name)(where=predicate)
                    )
                except Exception:  # noqa: BLE001
                    # Expression *construction* failed -- the method needs
                    # positional arguments, or does not apply to this dtype.
                    # The subject of this sweep is compilation, so anything
                    # that never became an expression is simply out of scope.
                    break
                try:
                    sql = to_sql(expr)
                except (
                    com.UnsupportedOperationError,
                    com.OperationNotDefinedError,
                ):
                    continue  # a deliberate refusal, not a malformed string
                checked += 1
                try:
                    sqlglot.parse_one(sql, dialect="redshift")
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{dtype_name}.{name}(where={predicate}): {exc}")
    assert checked > 50, f"sweep degenerated to {checked} compilations"
    assert not failures, "unparsable SQL emitted:\n" + "\n".join(failures)


def test_string_literal_escaping_matches_what_the_warehouse_decodes():
    """VERIFIED 2026-09-24. This was the most severe open finding; it resolved
    in the PR's favour, and one half of it turned out to be a silent bug fix.

    Retargeting the dialect swapped sqlglot's escape rules:
    ``Redshift.Tokenizer.STRING_ESCAPES`` is ``["\\\\", "'"]`` where
    Postgres's is ``["'"]``. Every string literal the backend emits changed. On
    the warehouse, ``LENGTH('\\t')`` is 1 and ``'\\t' = CHR(9)`` is true:
    Redshift decodes backslash escapes, so the new spelling is the correct one.

    The consequence nobody predicted is on the regex operations. Measured::

        '1' ~ '\\\\d'  -> MATCH      (what this backend emits now)
        '1' ~ '\\d'   -> NO MATCH   (what it emitted under the Postgres dialect)
        'd' ~ '\\d'   -> MATCH      (i.e. it meant the literal letter d)

    So every ``re_search``/``re_replace``/``re_extract``/``like`` carrying a
    backslash class was quietly wrong before this change and is right after it.
    That is a wrong-answer bug fixed by accident, which is the strongest
    argument in this module for auditing a dialect swap along every axis rather
    than only the constructs that prompted it.
    """
    t = xo.table({"s": "string"}, name="t")
    # user literals
    assert to_sql(t.filter(t.s == "a'b").select("s")).endswith("= 'a\\'b'")
    assert to_sql(t.filter(t.s == "a\\b").select("s")).endswith("= 'a\\\\b'")
    # the whitespace literal .strip() emits, which no user wrote
    assert "TRIM(' \\t\\n\\r\\v\\f' FROM" in to_sql(t.select(o=t.s.strip()))
    # the four regex operands -- doubled, which is what the warehouse decodes
    assert "'\\\\d+'" in to_sql(t.select(o=t.s.re_search(r"\d+")))
    assert "'\\\\s'" in to_sql(t.select(o=t.s.re_replace(r"\s", "")))
    assert "'(\\\\w+)'" in to_sql(t.select(o=t.s.re_extract(r"(\w+)", 1)))
    assert "LIKE 'a\\\\b'" in to_sql(t.select(o=t.s.like(r"a\b")))


def test_no_hand_written_override_is_clobbered_by_simple_ops():
    """``__init_subclass__`` runs AFTER the class body and can delete an
    override silently.

    ``compilers/base.py:458`` does ``setattr(cls, f"visit_{op}", make_impl(...))``
    for every entry in ``SIMPLE_OPS``, after the class body has already bound
    the hand-written methods. So an op appearing in BOTH ``SIMPLE_OPS`` and a
    hand-written ``visit_*`` loses the hand-written one, with no error and no
    warning. ``UNSUPPORTED_OPS`` is applied later still and wins over both.

    This backend now carries 55 inherited ``SIMPLE_OPS`` entries and a dozen
    hand-written overrides, most of which exist to fix defects found in review.
    Nothing but this test stands between the next added spelling and the silent
    deletion of one of them.

    Reads the class body's own source rather than ``vars()``: every generated
    method is set on the class too, so class-dict membership cannot tell a
    hand-written method from a generated one. What distinguishes them is where
    the surviving object was defined.
    """
    source = pathlib.Path(inspect.getfile(RedshiftCompiler)).read_text()
    tree = ast.parse(source)
    class_def = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RedshiftCompiler"
    )
    written = set()
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("visit_"):
            written.add(node.name)
        elif isinstance(node, ast.Assign):  # e.g. visit_MultiQuantile = visit_Quantile
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("visit_"):
                    written.add(target.id)
    assert len(written) >= 12, f"source scan degenerated to {written}"

    clobbered = sorted(
        name
        for name in written
        if getattr(getattr(RedshiftCompiler, name), "__module__", None)
        != RedshiftCompiler.__module__
    )
    assert not clobbered, (
        "these hand-written overrides no longer resolve to the redshift "
        f"compiler module -- SIMPLE_OPS or UNSUPPORTED_OPS replaced them: "
        f"{clobbered}"
    )
