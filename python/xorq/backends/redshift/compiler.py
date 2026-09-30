from __future__ import annotations

from xorq.backends.postgres.compiler import PostgresCompiler


class RedshiftCompiler(PostgresCompiler):
    """Redshift compiles as PostgreSQL for now.

    The dialect is deliberately *not* retargeted to sqlglot's Redshift yet.
    Retargeting looks free and is not: sqlglot's ``Redshift`` builds its
    ``TRANSFORMS`` by inheriting from ``Postgres`` at class-creation time, while
    ``xorq.vendor.ibis.backends.sql.dialects`` mutates
    ``Postgres.Generator.TRANSFORMS`` in place afterwards, so which one wins
    depends on import order. Retargeting therefore has to arrive together with
    an explicit ``TRANSFORMS`` and with unsupported operations raising, rather
    than compiling to structurally invalid SQL. That is its own change.

    Redshift is PostgreSQL-derived, so the postgres dialect is the correct
    conservative default in the meantime.
    """

    __slots__ = ()


compiler = RedshiftCompiler()
