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

    # Redshift-only type names as ``svv_all_columns`` actually reports them,
    # measured on a live warehouse (2026-09-24) rather than taken from the DDL
    # keyword: a ``VARBYTE(16)`` column comes back as **``binary varying``**,
    # because the view definition rewrites it. ``varbyte`` is kept beside it
    # because that is the name a user writes and may pass in by hand; the
    # spelling the catalog emits is the one that has to be here for the message
    # to name Redshift rather than falling through to the generic branch below.
    # ``varbinary`` is deliberately absent -- the inherited mapper maps it to
    # ``dt.Binary``, and no Redshift path produces it.
    _REDSHIFT_ONLY_TYPES = frozenset(
        {
            "super",
            "varbyte",
            "binary varying",
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

    @staticmethod
    def _unmappable_part(dtype: dt.DataType) -> dt.DataType | None:
        """The first component of ``dtype`` this backend cannot emit SQL for.

        Checking only the top-level type is not enough, and the gap is
        reachable: ``bpchar[]`` is a real spelling (psycopg names OID 1014
        exactly that, and ``svv_all_columns`` shows ``"char"[]``/``integer[]``
        for ``pg_catalog`` relations), and it maps to
        ``Array(value_type=Unknown)`` -- an ``Unknown`` the top-level check
        walks straight past.

        ``GeoSpatial`` is rejected alongside ``Unknown`` for the reason in the
        class docstring, and doing it structurally rather than by name also
        catches ``point``/``line``/``polygon``, which reach a geo type through
        ``PostgresType.unknown_type_strings`` and so never touch the name list.
        """
        if isinstance(dtype, (dt.Unknown, dt.GeoSpatial)):
            return dtype
        parts = ()
        if isinstance(dtype, dt.Array):
            parts = (dtype.value_type,)
        elif isinstance(dtype, dt.Map):
            parts = (dtype.key_type, dtype.value_type)
        elif isinstance(dtype, dt.Struct):
            parts = tuple(dtype.types)
        for part in parts:
            if (found := RedshiftType._unmappable_part(part)) is not None:
                return found
        return None

    @classmethod
    def from_string(cls, text: str, nullable: bool | None = None) -> dt.DataType:
        base, _, _ = (text or "").strip().partition("(")
        # Strip any array suffix before the alias lookup: the aliases are keyed
        # on the element spelling, and ``bpchar[]`` must reach the ``bpchar``
        # entry rather than missing it and degrading to an unknown element.
        element = base.strip().rstrip("[]").strip()
        lowered = element.lower()

        if lowered in cls._REDSHIFT_ONLY_TYPES:
            raise exc.UnsupportedBackendType(
                f"redshift type {base.strip()!r} has no xorq equivalent; it is a "
                f"Redshift-specific type this backend cannot map"
            )

        if (alias := cls._TYPE_ALIASES.get(lowered)) is not None:
            text = text.replace(element, alias, 1)

        try:
            dtype = super().from_string(text, nullable=nullable)
        except AttributeError as e:
            # The upstream mapper's own failure, re-raised as something that
            # names the type it choked on.
            raise exc.UnsupportedBackendType(
                f"redshift type {text!r} could not be mapped by the postgres "
                f"type mapper"
            ) from e

        if (bad := cls._unmappable_part(dtype)) is not None:
            raise exc.UnsupportedBackendType(
                f"redshift type {text!r} has no xorq equivalent (resolved to "
                f"{bad!r}); mapping it would hand back a schema that looks fine "
                f"and is wrong"
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
