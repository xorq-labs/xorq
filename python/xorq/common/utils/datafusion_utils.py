from __future__ import annotations

import contextlib
from typing import Any, Callable

import sqlglot as sg


def table_kind(default_schema: Callable[[], Any], name: str) -> str | None:
    """The kind of table ``name`` in the schema ``default_schema()`` returns.

    ``"physical"``, ``"view"`` or ``"temporary"`` (a ``TeeTableProvider``);
    ``None`` when there is no such table or the schema cannot be looked up (a
    session whose default catalog is not the hosts' ``datafusion``: neither
    host exposes the configured default outside information_schema, so a
    temporary table there is left to ``DROP``, which keeps it registered). The
    hosts name the default schema differently: ``ctx.catalog().database()`` on
    xorq-datafusion, ``ctx.catalog().schema()`` on datafusion-python.
    """
    with contextlib.suppress(Exception):
        return default_schema().table(name).kind
    return None


def deregister_temporary_table(
    ctx: Any, default_schema: Callable[[], Any], name: str, quoted: bool
) -> bool:
    """Deregister ``name`` if it is a temporary table in the session's default
    schema.

    DataFusion's ``DROP TABLE`` skips a temporary table (a ``TeeTableProvider``
    is one): it no-ops with ``IF EXISTS`` and raises "doesn't exist" without
    it. Base tables and views are left to ``DROP``. Returns whether the table
    was deregistered.
    """
    if table_kind(default_schema, name) != "temporary":
        return False
    ctx.deregister_table(str(sg.to_identifier(name, quoted=quoted)))
    return True
