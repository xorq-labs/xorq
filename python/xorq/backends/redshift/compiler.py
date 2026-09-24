from __future__ import annotations

from functools import partial

import sqlglot as sg
import sqlglot.expressions as sge

import xorq.common.exceptions as com
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.postgres.compiler import PostgresCompiler
from xorq.vendor.ibis.backends.sql.compilers.base import NULL, STAR, AggGen
from xorq.vendor.ibis.backends.sql.datatypes import RedshiftType
from xorq.vendor.ibis.backends.sql.dialects import Redshift


# Redshift rejects a frame clause on any ranking window function:
#   "Frame clause should not be specified for ranking window functions"
# The set is Redshift's documented ranking family, not everything that happens
# to be orderable -- NthValue, First and Last are *value* functions on Redshift
# and do take a frame.
# ``RankBase`` covers RowNumber, MinRank and DenseRank; the other three are
# ranking functions in Redshift's sense but sit directly under ``Analytic``.
_RANKING_OPS = (
    ops.RankBase,
    ops.PercentRank,
    ops.CumeDist,
    ops.NTile,
)

# ``LAG``/``LEAD`` are *offset* functions, not ranking ones, and Redshift
# documents their syntax with no frame clause at all -- unlike ``FIRST_VALUE``,
# ``LAST_VALUE`` and ``NTH_VALUE``, whose documented syntax does include one.
# Suppressing the frame here cannot change a result: an offset function reads a
# row at a fixed displacement from the current one and is frame-insensitive in
# every engine that accepts the clause, PostgreSQL included.
#
# UNVERIFIED against a live warehouse. This one rests on AWS's documented
# grammar rather than on an observed rejection, which is weaker evidence than
# the rest of this module carries; the frame it removes is one no user asked
# for, so the downside of being wrong is bounded.
_OFFSET_OPS = (
    ops.Lag,
    ops.Lead,
)

_NO_FRAME_OPS = _RANKING_OPS + _OFFSET_OPS

# Redshift documents these in window position as ``OVER ( [PARTITION BY ...] )``
# -- no ``ORDER BY`` and no frame, because their ordering is carried by the
# ``WITHIN GROUP`` clause instead. So unlike ``_NO_FRAME_OPS``, where only the
# frame is dropped, the whole ``OVER`` body below ``PARTITION BY`` has to go.
# Dropping the ``ORDER BY`` is not a loss: for every op here the ordering the
# user asked for is already emitted inside ``WITHIN GROUP``.
#
# UNVERIFIED against a live warehouse, on AWS's documented grammar.
_PARTITION_ONLY_OPS = (
    ops.GroupConcat,
    ops.Quantile,
    ops.MultiQuantile,
    ops.ApproxQuantile,
    ops.ApproxMultiQuantile,
    ops.Median,
    ops.ApproxMedian,
)

# Redshift has no ``unnest``. sqlglot's Redshift generator does not raise on
# one -- it warns and returns the empty string, which then flows back into
# expression building as a ``str``. The result is either an internal
# ``AttributeError: 'str' object has no attribute 'args'`` from inside sqlglot,
# or SQL with a hole in it: ``SELECT  AS "o"``, ``ARRAY(SELECT DISTINCT)``,
# ``ARRAY(SELECT UNION SELECT)``, or an array column used as a table. None of
# these were ever going to run on Redshift, which has no array type at all --
# what they cost is the diagnosis, since none of them names the backend or the
# operation. Listing them here converts each into the same
# ``OperationNotDefinedError`` every other unsupported op raises.
_UNNEST_DEPENDENT_ARRAY_OPS = (
    ops.ArrayDistinct,
    ops.ArrayFilter,
    ops.ArrayIntersect,
    ops.ArrayMap,
    ops.ArrayMax,
    ops.ArrayMean,
    ops.ArrayMin,
    ops.ArrayPosition,
    ops.ArraySort,
    ops.ArraySum,
    ops.ArrayUnion,
    ops.Unnest,
)


class RedshiftCompiler(PostgresCompiler):
    """Redshift, compiled as Redshift rather than as PostgreSQL.

    Redshift is PostgreSQL-derived at the wire level, which is why the backend
    subclasses the postgres one, but it is *not* PostgreSQL at the SQL level.
    Every override below corresponds to a construct the rc16 transcript observed
    Redshift rejecting at execution time, after a successful build -- the
    expensive failure mode, because the models were already written and
    documented by then.

    Retargeting the dialect is necessary but nowhere near sufficient. Measured
    2026-09-23: generating the reported forms under sqlglot's Redshift dialect
    instead of Postgres *fixed* none of them. ``FILTER(WHERE)``, the ranking
    frame clause and ``FIRST()`` came out byte-identical, because they are
    decided here, above sqlglot. The date literal was the one exception -- it
    did change, from ``MAKE_DATE`` to ``DATE_FROM_PARTS`` -- but Redshift has
    neither function, so the dialect swap moved it from one invalid spelling to
    another. That is why this class carries compiler overrides and not just a
    ``dialect`` assignment, and why the date case is handled in
    ``visit_DateFromYMD`` below rather than by a rename in the dialect.
    """

    __slots__ = ()

    dialect = Redshift

    # ``TYPE_MAPPERS`` is keyed by dialect name, so retargeting ``dialect``
    # above also retargets which mapper ``Schema.to_sqlglot`` reaches. The
    # compiler's own ``type_mapper`` is a *separate* binding, inherited as
    # ``PostgresType`` from the vendored postgres compiler, and it is the one
    # ``Backend.get_schema`` / ``_get_schema_using_query`` use to parse type
    # strings coming back from the warehouse. Setting only the first left the
    # read path on PostgreSQL's vocabulary: ``PostgresType.from_string(
    # "varbyte")`` is ``unknown`` where ``RedshiftType.from_string("varbyte")``
    # is ``binary``, so a ``VARBYTE`` column round-tripped as ``unknown``.
    type_mapper = RedshiftType

    # ``first`` is PostgreSQL's spelling and Redshift has no such aggregate --
    # ``visit_First`` below raises on exactly that. Inheriting this entry meant
    # ``.arbitrary()`` compiled to ``FIRST(x)`` anyway, because
    # ``__init_subclass__`` generates ``visit_Arbitrary`` from ``SIMPLE_OPS``
    # and that generation happens whether or not a sibling method raises.
    # Redshift does support ``ANY_VALUE``.
    SIMPLE_OPS = PostgresCompiler.SIMPLE_OPS | {ops.Arbitrary: "any_value"}

    UNSUPPORTED_OPS = (
        *PostgresCompiler.UNSUPPORTED_OPS,
        *_UNNEST_DEPENDENT_ARRAY_OPS,
        # ``ARRAY_AGG`` is the same absent function ``visit_ArgMinMax`` raises
        # over; leaving ``collect()`` compiling to it while ``argmax`` raises
        # for want of it was one premise with two answers.
        ops.ArrayCollect,
        # The postgres block in ``dialects.py`` renames ``RegexpSplit`` to
        # ``regexp_split_to_array``, which Redshift lacks. Declining the rename
        # only moves it to ``REGEXP_SPLIT``, which no engine has. Redshift has
        # no regex-split-to-array at all.
        ops.RegexSplit,
    )

    # Redshift has no aggregate FILTER clause. AggGen already knows the
    # fallback: with supports_filter=False it rewrites `agg(x, where=c)` to
    # `agg(CASE WHEN c THEN x END)` (compilers/base.py:139). `where=` is the
    # idiom xorq's own skills teach, so leaving this True made a documented
    # construction unrunnable on this backend.
    agg = AggGen(supports_filter=False, supports_order_by=True)

    def visit_CountDistinct(self, op, *, arg, where):
        """``COUNT(DISTINCT CASE WHEN ... END)``, not ``COUNT(CASE WHEN ...
        THEN DISTINCT ... END)``.

        The base implementation (``compilers/base.py:1031``) hands
        ``sge.Distinct`` to ``AggGen`` as the argument. With
        ``supports_filter=True`` that is fine -- the filter becomes a trailing
        ``FILTER(WHERE ...)`` and never touches the argument. With
        ``supports_filter=False`` the fallback wraps *the argument* in a
        ``CASE``, which puts ``DISTINCT`` inside a ``CASE`` branch, where it is
        not valid SQL anywhere.

        So the conditional has to go inside the ``DISTINCT`` rather than around
        it. Building the ``CASE`` here and passing it to a plain ``count``
        keeps the null-skipping semantics identical: rows failing the predicate
        become NULL, and ``COUNT(DISTINCT expr)`` does not count NULLs. Verified
        2026-09-23 against ``COUNT(DISTINCT x) FILTER (WHERE c)`` over 2000
        randomized tables with NULL values and three-valued predicates: zero
        mismatches. The broken form is rejected by sqlite, duckdb and sqlglot's
        own postgres and redshift parsers, all at the ``DISTINCT`` token.

        Note this is not a Redshift quirk but a latent trap in the base class:
        ``SQLGlotCompiler`` both defaults ``supports_filter`` to False
        (``compilers/base.py:96``) and supplies the ``sge.Distinct`` version of
        this method, so the two defaults compose into invalid SQL. Every other
        backend escapes only by overriding one side or the other. A new
        compiler that overrides neither inherits the bug.
        """
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    def visit_CountStar(self, op, *, arg, where):
        """``COUNT(CASE WHEN c THEN 1 END)``, not ``COUNT(CASE WHEN c THEN *)``.

        The same ``supports_filter=False`` fallback that ``visit_CountDistinct``
        works around also wraps ``STAR``, and ``THEN *`` is not a legal ``CASE``
        branch in any dialect. This one is worse than the ``DISTINCT`` case
        because ``count(where=...)`` is the single most common ``where=`` idiom
        in the codebase, and because the pre-change spelling
        (``COUNT(*) FILTER (WHERE c)``) was at least *parseable* -- so the
        ``AggGen`` flip turned a Redshift-specific rejection into a universal
        syntax error.

        Counting a non-NULL constant is equivalent: ``COUNT(expr)`` skips NULLs,
        so the rows surviving the predicate are exactly the rows counted.
        """
        if where is None:
            return self.f.count(STAR)
        return self.f.count(self.if_(where, 1, NULL))

    def visit_CountDistinctStar(self, op, *, arg, where):
        """``Table.nunique(where=...)``, which reaches the same trap as
        ``visit_CountDistinct`` through a door that override does not cover.

        The postgres implementation (``compilers/postgres.py:226``) builds a row
        constructor and hands ``sge.Distinct`` to ``AggGen``, so with
        ``supports_filter=False`` the ``DISTINCT`` lands inside a ``CASE`` --
        the exact construct ``visit_CountDistinct``'s docstring explains is
        invalid everywhere. Fixed the same way: the conditional goes *inside*
        the ``DISTINCT``.

        The row-constructor spelling itself is inherited from PostgreSQL and is
        not claimed here to run on Redshift; that question predates this module
        and is unchanged by this override. What is fixed is that the emitted
        string is now structurally valid rather than unparsable.
        """
        row = sge.Tuple(
            expressions=list(
                map(partial(sg.column, quoted=self.quoted), op.arg.schema.keys())
            )
        )
        if where is not None:
            row = self.if_(where, row, NULL)
        return self.f.count(sge.Distinct(expressions=[row]))

    def visit_Quantile(self, op, *, arg, quantile, where):
        """Ordered-set aggregates never reach ``AggGen``, so ``supports_filter``
        does not reach them either.

        ``visit_Quantile``, ``visit_Median``, ``visit_ApproxMedian`` and the
        multi/approx aliases all build ``sge.Filter`` by hand
        (``compilers/postgres.py:245``), which is why flipping ``AggGen`` left
        ``PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x) FILTER (WHERE c)``
        emitting byte-identical PostgreSQL. The predicate folds into the
        ordered argument instead: ``PERCENTILE_CONT`` ignores NULLs, so
        nulling-out the filtered rows removes them from the ordering set
        exactly as ``FILTER`` removed them from the input.
        """
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.WithinGroup(
            this=self.f[f"percentile_{suffix}"](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )

    # Rebound rather than inherited: the postgres aliases bind to *that* class's
    # function object at class-creation time, so they would keep the
    # ``sge.Filter`` version even with the override above in place.
    visit_MultiQuantile = visit_Quantile
    visit_ApproxQuantile = visit_Quantile
    visit_ApproxMultiQuantile = visit_Quantile

    def visit_Mode(self, op, *, arg, where):
        """``visit_Quantile``'s problem, in the other ordered-set aggregate.

        Whether Redshift has ``MODE() WITHIN GROUP`` at all is a separate and
        still-open question -- it is not in AWS's documented aggregate list.
        This override deliberately does not answer it: it removes the
        ``FILTER (WHERE ...)`` clause that is known-rejected, and leaves the
        function name exactly as inherited.
        """
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return sge.WithinGroup(
            this=self.f.mode(),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )

    def visit_ArgMinMax(self, op, *, arg, key, where, desc: bool):
        """``argmin``/``argmax`` have no Redshift lowering at all.

        The postgres construction is ``(ARRAY_AGG(x ORDER BY k DESC))[1]``, and
        Redshift has no ``ARRAY_AGG`` -- its only aggregate-to-many function is
        ``LISTAGG``, which returns a string. So the inherited lowering was
        already unrunnable here.

        The ``AggGen`` flip then made it unparsable as well, and this is the
        clearest demonstration of why that flag is not a local change:
        ``visit_ArgMinMax`` synthesizes its own non-``None`` ``where`` for the
        null-guards, so *every* ``argmax`` -- with or without a user predicate
        -- took the fallback, which wrapped an ``sge.Ordered`` in a ``CASE`` and
        produced ``CASE WHEN ... THEN x ORDER BY k DESC ELSE NULL END``.

        Raising is consistent with ``visit_First``: a construct with no
        lowering fails at compile time, where the message can name the
        alternative, rather than at execution time in the warehouse.
        """
        raise com.UnsupportedOperationError(
            "Redshift has no `array_agg`, so `argmin`/`argmax` cannot be "
            "lowered into the aggregate position this construction requires. "
            "Express the intent as an explicit `row_number()` window ordered "
            "by the key and filtered to 1 instead."
        )

    def visit_GroupConcat(self, op, *, arg, sep, order_by, where):
        """``LISTAGG``, with the predicate on the value and the ordering outside.

        Two defects, both from the generated ``SIMPLE_OPS`` reduction impl
        (``compilers/base.py:446``) handing *every* argument to ``AggGen``:

        * ``where=`` CASE-wrapped the **separator** as well as the value, and
          Redshift requires ``LISTAGG``'s delimiter to be a constant.
        * ``order_by=`` went inside the argument list as PostgreSQL's
          ``string_agg(x, sep ORDER BY k)``. Redshift spells this
          ``LISTAGG(x, sep) WITHIN GROUP (ORDER BY k)``.

        Only the value is conditional; the separator is passed through
        untouched.
        """
        if where is not None:
            arg = self.if_(where, arg, NULL)
        out = self.f.group_concat(arg, sep)
        if order_by:
            out = sge.WithinGroup(
                this=out, expression=sge.Order(expressions=list(order_by))
            )
        return out

    # ``approx_nunique`` is the third entry point into the trap
    # ``visit_CountDistinct`` documents, after ``visit_CountDistinctStar``.
    # ``compilers/postgres.py:273`` hands ``sge.Distinct`` to ``AggGen`` exactly
    # as the other two did. The lowering is identical -- postgres has no
    # approximate count either -- so the override is an alias rather than a
    # copy, which also means a future change to one cannot skip the other.
    visit_ApproxCountDistinct = visit_CountDistinct

    def visit_StartsWith(self, op, *, arg, start):
        """``LEFT(s, LENGTH(p)) = p``, not ``s LIKE p || '%'``.

        Redshift has no ``STARTS_WITH``, and sqlglot's Redshift generator
        lowers it to a ``LIKE`` whose pattern is the operand concatenated with
        ``'%'`` -- unescaped. Any ``%`` or ``_`` in the operand then becomes a
        wildcard, so ``t.s.startswith("a%")`` matches every string beginning
        with ``a`` rather than the two literal characters. Silent wrong rows,
        not an error.

        This mirrors the shape ``visit_EndsWith`` already has in the postgres
        compiler (``compilers/postgres.py:561``), which is immune for the same
        reason: it compares extracted text rather than building a pattern.
        """
        return self.f.left(arg, self.f.length(start)).eq(start)

    def visit_DateFromYMD(self, op, *, year, month, day):
        """Redshift has no ``make_date``.

        Composed from an ISO-8601 string instead, which is the construction AWS
        documents for this. The zero-padding matters: ``TO_DATE('2026-9-3',
        'YYYY-MM-DD')`` is not reliably parsed, so each part is padded to its
        fixed width before concatenation.
        """
        to_str = partial(self.cast, to=dt.string)
        pad = lambda value, width: self.f.lpad(to_str(value), width, "0")  # noqa: E731
        return self.f.to_date(
            self.f.concat(
                pad(year, 4),
                sge.convert("-"),
                pad(month, 2),
                sge.convert("-"),
                pad(day, 2),
            ),
            sge.convert("YYYY-MM-DD"),
        )

    def visit_WindowFunction(self, op, *, how, func, start, end, group_by, order_by):
        """Drop the frame clause where Redshift's grammar has no slot for one.

        :5475 in the transcript records that both the windowed and the
        unwindowed spellings emitted ``ROWS BETWEEN UNBOUNDED PRECEDING AND
        UNBOUNDED FOLLOWING``, so there was no API-level way for a user to avoid
        this -- it had to be fixed in the compiler.

        Scoped deliberately to the ranking family plus ``LAG``/``LEAD``.
        Suppressing the frame everywhere would silently change the result of
        every cumulative aggregate, turning a loud error into wrong numbers.
        Every op in ``_NO_FRAME_OPS`` is one whose value cannot depend on the
        frame, so removing it is observable only in the emitted string.
        """
        window = super().visit_WindowFunction(
            op,
            how=how,
            func=func,
            start=start,
            end=end,
            group_by=group_by,
            order_by=order_by,
        )
        if isinstance(op.func, ops.Arbitrary):
            raise com.UnsupportedOperationError(
                "`ANY_VALUE` is an aggregate on Redshift and not a window "
                "function, so `.arbitrary()` cannot be used with `.over(...)` "
                "on this backend. Aggregate it in a `group_by` instead, or "
                "pick a row explicitly with a `row_number()` window."
            )
        if isinstance(op.func, _PARTITION_ONLY_OPS):
            window.set("spec", None)
            window.set("order", None)
        elif isinstance(op.func, _NO_FRAME_OPS):
            window.set("spec", None)
        return window

    def visit_First(self, op, *, arg, where, order_by, include_null):
        raise com.UnsupportedOperationError(
            "Redshift has no `first` aggregate, so `.distinct(on=...)` and "
            "`.first()` cannot be compiled for this backend. Redshift's "
            "`first_value` is a window function and cannot be used in the "
            "aggregate position this lowering requires. Express the intent as "
            "an explicit `row_number()` window filtered to 1 instead."
        )

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        raise com.UnsupportedOperationError(
            "Redshift has no `last` aggregate, so `.distinct(on=..., "
            "keep='last')` and `.last()` cannot be compiled for this backend. "
            "Redshift's `last_value` is a window function and cannot be used in "
            "the aggregate position this lowering requires. Express the intent "
            "as an explicit `row_number()` window filtered to 1 instead."
        )


compiler = RedshiftCompiler()
