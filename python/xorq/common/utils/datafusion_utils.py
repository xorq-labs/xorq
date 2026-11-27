from __future__ import annotations

import contextlib
from typing import Any, Callable

import sqlglot as sg


def table_kind(get_schema: Callable[[], Any], name: str) -> str | None:
    """The kind of table ``name`` in the schema ``get_schema()`` returns.

    ``"physical"``, ``"view"`` or ``"temporary"`` (a ``TeeTableProvider``);
    ``None`` when there is no such table or the schema cannot be looked up
    (on datafusion-python, a session whose default catalog is not
    ``datafusion``: a temporary table there is left to ``DROP``, which keeps
    it registered). xorq-datafusion resolves the configured default catalog
    and schema; datafusion-python uses ``ctx.catalog().schema()``.
    """
    with contextlib.suppress(Exception):
        return get_schema().table(name).kind
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
