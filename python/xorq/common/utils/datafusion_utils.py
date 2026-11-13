from __future__ import annotations

import contextlib
from typing import Any

import sqlglot as sg


def table_kind(schema: Any, name: str) -> str | None:
    """The kind of table ``name`` in a catalog ``schema`` object.

    ``"physical"``, ``"view"`` or ``"temporary"`` (a ``TeeTableProvider``);
    ``None`` when there is no such table. The hosts name the default schema
    differently: ``ctx.catalog().database()`` on xorq-datafusion,
    ``ctx.catalog().schema()`` on datafusion-python.
    """
    with contextlib.suppress(Exception):
        return schema.table(name).kind
    return None


def deregister_temporary_table(ctx: Any, schema: Any, name: str, quoted: bool) -> bool:
    """Deregister ``name`` if it is a temporary table in ``schema``, the
    session's default schema.

    DataFusion's ``DROP TABLE`` skips a temporary table (a ``TeeTableProvider``
    is one): it no-ops with ``IF EXISTS`` and raises "doesn't exist" without
    it. Base tables and views are left to ``DROP``. Returns whether the table
    was deregistered.
    """
    if table_kind(schema, name) != "temporary":
        return False
    ctx.deregister_table(str(sg.to_identifier(name, quoted=quoted)))
    return True
