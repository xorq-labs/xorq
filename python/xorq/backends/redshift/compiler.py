from __future__ import annotations

import xorq.common.exceptions as exc
from xorq.backends.postgres.compiler import PostgresCompiler
from xorq.vendor.ibis.backends.sql.datatypes import PostgresType
from xorq.vendor.ibis.expr import datatypes as dt


class RedshiftType(PostgresType):
    """PostgreSQL's type mapper, made loud about what Redshift adds to it.

    ``PostgresType.from_string`` has three behaviours this backend cannot
    live with, and they are shared by *both* introspection paths -- the
    ``svv_all_columns`` catalog read and the ``cursor.description`` probe --
    which is why the repair lives in the mapper rather than in either caller:

    1. An unparsable type string falls back to ``dt.unknown`` **and drops the
       ``nullable=`` argument on the way** (``SqlglotType.from_string``), so a
       ``SUPER NOT NULL`` column comes back with the wrong type *and* the
       wrong nullability, and nothing says so.
    2. ``geometry`` and ``geography`` parse cleanly into ibis ``GeoSpatial``
       types. That is worse than ``unknown``: this compiler is still the
       PostgreSQL one, so a geo operation on such a column compiles to a
       PostGIS call Redshift does not implement -- a server-side error on SQL
       that looked fine, which is the exact failure class this backend's
       overrides exist to eliminate.
    3. A handful of OIDs (``oid``, ``regclass``, ``regproc`` and friends)
       make the upstream mapper raise a bare
       ``AttributeError: 'str' object has no attribute 'name'`` from inside
       ``to_ibis``, which names neither the column nor the type.

    The policy is the one the query path's docstring already stated and only
    half-implemented: a type this backend cannot map raises
    ``UnsupportedBackendType`` rather than degrading to a schema that looks
    fine and is wrong.
    """

    # Redshift-only type names that ``svv_all_columns`` reports. ``geometry``
    # and ``geography`` are listed because they *do* parse -- see the class
    # docstring -- so leaving them out would let them through silently.
    _REDSHIFT_ONLY_TYPES = frozenset(
        {
            "super",
            "varbyte",
            "varbinary",
            "hllsketch",
            "geometry",
            "geography",
        }
    )

    # Spellings that mean a character type but that sqlglot's postgres dialect
    # does not parse. ``bpchar`` is not exotic: it is what PostgreSQL -- and so
    # psycopg's OID registry, at OID 1042 -- calls every ``CHAR(n)`` column, so
    # without this entry the *query* path types every ``CHAR`` column
    # ``unknown`` while the *catalog* path (which sees ``character``) types the
    # same column ``string``.
    _TYPE_ALIASES = {
        "bpchar": "character",
        '"char"': "character",
    }

    @classmethod
    def from_string(cls, text: str, nullable: bool | None = None) -> dt.DataType:
        base, _, _ = (text or "").strip().partition("(")
        lowered = base.strip().lower()

        if lowered in cls._REDSHIFT_ONLY_TYPES:
            raise exc.UnsupportedBackendType(
                f"redshift type {base.strip()!r} has no xorq equivalent; it is a "
                f"Redshift-specific type this backend cannot map"
            )

        if (alias := cls._TYPE_ALIASES.get(lowered)) is not None:
            text = text.replace(base.strip(), alias, 1)

        try:
            dtype = super().from_string(text, nullable=nullable)
        except AttributeError as e:
            # The upstream mapper's own failure, re-raised as something that
            # names the type it choked on.
            raise exc.UnsupportedBackendType(
                f"redshift type {text!r} could not be mapped by the postgres "
                f"type mapper"
            ) from e

        if isinstance(dtype, dt.Unknown):
            raise exc.UnsupportedBackendType(
                f"redshift type {text!r} has no xorq equivalent; mapping it to "
                f"'unknown' would hand back a schema that looks fine and is wrong"
            )

        # ``unknown_type_strings`` hits and the ``dt.unknown`` fallback both
        # return a dtype built with the mapper's *default* nullability, ignoring
        # the argument. Nothing above can reach the fallback any more, but the
        # copy is cheap and makes the postcondition unconditional.
        if nullable is not None and dtype.nullable != nullable:
            dtype = dtype.copy(nullable=nullable)
        return dtype


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

    The *type mapper* is retargeted, which is a separate question from the
    dialect: it decides what a catalog row or a result description means, not
    what SQL is generated, so it can be made Redshift-aware without touching
    the generator. See ``RedshiftType``.
    """

    __slots__ = ()

    type_mapper = RedshiftType


compiler = RedshiftCompiler()
