from __future__ import annotations

from functools import partial

import sqlglot.expressions as sge

import xorq.common.exceptions as com
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.postgres.compiler import PostgresCompiler
from xorq.vendor.ibis.backends.sql.compilers.base import NULL, AggGen
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
        """Drop the frame clause for ranking functions only.

        :5475 in the transcript records that both the windowed and the
        unwindowed spellings emitted ``ROWS BETWEEN UNBOUNDED PRECEDING AND
        UNBOUNDED FOLLOWING``, so there was no API-level way for a user to avoid
        this -- it had to be fixed in the compiler.

        Scoped to ranking functions deliberately. Suppressing the frame
        everywhere would silently change the result of every cumulative
        aggregate, turning a loud error into wrong numbers.
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
        if isinstance(op.func, _RANKING_OPS):
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
