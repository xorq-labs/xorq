"""Live-Redshift probe harness: SELECT-only, table-anchored, env-gated.

Shared by the conftest beside it and the live tests; kept out of conftest.py
because several backends have one, so ``import conftest`` is ambiguous.

Every test under this directory is marked ``redshift``, which no CI job
selects, and each one skips unless the ``XORQ_REDSHIFT_*`` variables name a
warehouse. Nothing here is required for a pull request to go green.

The harness exists because hand-written probe scripts kept answering the wrong
question. Three rules are therefore enforced rather than remembered:

* **A probe must read a table.** A query that reads none is answered by
  Redshift's leader node, which accepts more than the compute nodes do:
  ``'ab' = 'ab '`` is false there and true on compute, and ``CAST(... AS INET)``
  is accepted there and rejected on compute. A leader-node rejection is still
  evidence of absence, which is why ``redshift_evidence`` may record one; a
  probe here must reach compute, so a statement with no ``FROM schema.table``
  is refused before it is sent.
* **A probe is SELECT-only**, checked before it is sent. The check is a
  denylist over the statement text -- mutating keywords, ``SELECT ... INTO``
  (which creates a table), and the system functions with side effects -- so it
  refuses what it names and nothing else. It is the only control when the
  runner connects as an admin user, which the one runner in use does.
* **One statement per test**, on an autocommit connection, so one rejection
  cannot mask the next.

``XORQ_REDSHIFT_PROBE_TABLE`` names a ``schema.table`` with at least two rows
and the columns ``id`` (bigint, unique), ``title`` (varchar), ``is_live``
(boolean) and ``fee_rate`` (numeric).
"""

from __future__ import annotations

import re

from xorq.common.utils.env_utils import EnvConfigable


RedshiftLiveConfig = EnvConfigable.subclass_from_kwargs(
    "XORQ_REDSHIFT_HOST",
    "XORQ_REDSHIFT_USER",
    "XORQ_REDSHIFT_PASSWORD",
    "XORQ_REDSHIFT_PROBE_TABLE",
    XORQ_REDSHIFT_PORT="5439",
    XORQ_REDSHIFT_DATABASE="dev",
)

REQUIRED = (
    "XORQ_REDSHIFT_HOST",
    "XORQ_REDSHIFT_USER",
    "XORQ_REDSHIFT_PASSWORD",
    "XORQ_REDSHIFT_PROBE_TABLE",
)

PROBE_TABLE_SCHEMA = {
    "id": "int64",
    "title": "string",
    "is_live": "boolean",
    "fee_rate": "decimal(18, 4)",
}

_READS_A_TABLE = re.compile(r'\bFROM\s+"?\w+"?\s*\.\s*"?\w+"?', re.IGNORECASE)
_SELECT_ONLY = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_MUTATING = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|"
    r"COPY|UNLOAD|VACUUM|ANALYZE|SET|CALL|INTO)\b",
    re.IGNORECASE,
)
# Functions a SELECT can call for their side effect: ending or cancelling
# another session, and changing a setting.
_SIDE_EFFECTING = re.compile(
    r"\b(PG_TERMINATE_BACKEND|PG_CANCEL_BACKEND|SET_CONFIG)\s*\(",
    re.IGNORECASE,
)


def refuse_unless_compute_select(sql: str) -> None:
    """Raise unless ``sql`` is one read-only statement that reads a table."""
    if not _SELECT_ONLY.match(sql) or ";" in sql.rstrip().rstrip(";"):
        raise ValueError(f"not a single SELECT: {sql}")
    # Redshift string literals take backslash escapes as well as a doubled
    # quote, so both are consumed inside a literal: ``'a\''`` is one literal.
    unquoted = re.sub(r"'(?:[^'\\]|''|\\.)*'", "''", sql)
    if _MUTATING.search(unquoted):
        raise ValueError(f"refusing a statement with a mutating keyword: {sql}")
    if _SIDE_EFFECTING.search(unquoted):
        raise ValueError(f"refusing a call made for its side effect: {sql}")
    if not _READS_A_TABLE.search(unquoted):
        raise ValueError(
            "refusing a probe that reads no table: the leader node would answer "
            f"it, and its acceptances do not hold on compute: {sql}"
        )
