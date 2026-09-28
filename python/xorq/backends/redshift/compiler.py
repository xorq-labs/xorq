from __future__ import annotations

import xorq.common.exceptions as exc
from xorq.backends.postgres.compiler import PostgresCompiler
from xorq.vendor.ibis.backends.sql.datatypes import PostgresType
from xorq.vendor.ibis.common.annotations import ValidationError
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

    # Redshift-only type names with no xorq equivalent. ``VARBYTE`` is not
    # one: it is variable-length binary data, and maps to ``dt.Binary`` through
    # ``_TYPE_ALIASES`` below.
    #
    # The two interval column types cannot be *read* as ibis intervals: a live
    # result description carries them as OIDs 1188/1190, which psycopg has no
    # loader for, so each value arrives as Redshift's text rendering
    # (``'1 mon'``). ``svv_columns`` spells them ``intervaly2m`` and
    # ``intervald2s`` (measured), which sqlglot does not parse; the DDL
    # spellings are listed too because they *do* parse, into intervals, and
    # ``svv_all_columns``' spelling for a permanent table is unmeasured.
    _REDSHIFT_ONLY_TYPES = frozenset(
        {
            "super",
            "hllsketch",
            "geometry",
            "geography",
            "intervaly2m",
            "intervald2s",
            "interval year to month",
            "interval day to second",
        }
    )

    # Spellings Redshift reports that sqlglot's postgres dialect does not
    # parse. ``bpchar`` is not exotic: it is what PostgreSQL -- and so
    # psycopg's OID registry, at OID 1042 -- calls every ``CHAR(n)`` column, so
    # without this entry the *query* path types every ``CHAR`` column
    # ``unknown`` while the *catalog* path (which sees ``character``) types the
    # same column ``string``.
    #
    # ``VARBYTE`` has two spellings and sqlglot's postgres dialect parses
    # neither: ``varbyte`` is the DDL keyword a user writes, and ``binary
    # varying`` is what ``svv_all_columns`` reports for a ``VARBYTE(16)``
    # column, measured on a live warehouse (2026-09-24) -- the view definition
    # rewrites it. Both are sent to ``varbinary``, which the inherited mapper
    # maps to ``dt.Binary``.
    #
    # ``int8`` and ``float`` are spellings sqlglot *does* parse, but not as
    # Redshift means them on every supported version: at 23.6.3 ``int8`` is an
    # 8-bit integer and ``float`` a 32-bit one, where Redshift's are ``BIGINT``
    # and ``DOUBLE PRECISION``. ``int8`` is what psycopg names OID 20, so without
    # this every ``BIGINT`` through ``con.sql`` -- ``COUNT(*)`` included --
    # bound as ``int8`` there. ``int8`` is the only name in psycopg's registry
    # whose mapping differs between 23.6.3 and 28.6.0; ``float`` is not in the
    # registry, and is aliased because Redshift documents it as a synonym.
    _TYPE_ALIASES = {
        "bpchar": "character",
        '"char"': "character",
        "varbyte": "varbinary",
        "binary varying": "varbinary",
        "int8": "bigint",
        "float": "double precision",
    }

    @staticmethod
    def unmappable_part(dtype: dt.DataType) -> dt.DataType | None:
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
            if (found := RedshiftType.unmappable_part(part)) is not None:
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
        except (AttributeError, TypeError, ValueError, ValidationError) as e:
            # The upstream mapper's own failures, re-raised as something that
            # names the type it choked on, so the column binds as unknown
            # rather than failing its whole table. The ``AttributeError`` is
            # ``oid``'s; the rest are a type whose modifier the dtype rejects
            # -- ``interval(6)``, psycopg's name for an interval with a
            # precision, and ``numeric(0,0)`` or ``timestamp(10)``.
            raise exc.UnsupportedBackendType(
                f"redshift type {text!r} could not be mapped by the postgres "
                f"type mapper"
            ) from e

        if (bad := cls.unmappable_part(dtype)) is not None:
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
