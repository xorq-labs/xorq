"""What Redshift was measured to lack, as data the Redshift tests assert against.

One place for each measured fact, so that a docstring, a test and the compiler
cannot drift apart on it: they cite a key here instead of restating the result.

A measurement is admissible only when it would survive the leader-node trap. A
query that reads no table is answered by Redshift's leader node, which accepts
more than the compute nodes do, so a REJECTION there is evidence of absence and
an acceptance is not. Every entry below is a rejection, or a probe that read a
table.
"""

from __future__ import annotations

import re


# Function names, as the compiler spells them BEFORE the dialect renders them
# (``self.f.<name>``, ``self.agg.<name>``, or a ``SIMPLE_OPS`` target), that
# Redshift does not have. No op the Redshift compiler can compile may reach one;
# ``test_redshift_op_inventory.py`` enforces it over every inherited visitor.
MEASURED_ABSENT_FUNCTIONS: dict[str, str] = {
    "make_date": "rejected 2026-09-24",
    "datefromparts": "renders DATE_FROM_PARTS; rejected 2026-09-24",
    "make_time": "rejected 2026-09-28",
    "make_timestamp": "rejected on compute 2026-09-28",
    "date_bin": "rejected on compute 2026-09-28",
    "bit_xor": "rejected on compute 2026-09-28",
    "array_agg": "rejected 2026-09-24",
    "first": "rejected 2026-09-24",
    "mode": "rejected in both spellings 2026-09-24",
    "percentile_disc": "rejected 2026-09-24",
    "regexp_match": "rejected 2026-09-28",
    "hashtextextended": "rejected 2026-09-28",
    "gen_random_uuid": "rejected 2026-09-28",
    "encode": "rejected on compute 2026-09-28",
    "pg_typeof": "rejected on compute 2026-09-28",
    "array_remove": "rejected on compute 2026-09-28",
    "cardinality": "rejected on compute 2026-09-28",
    "corr": "rejected on compute for bigint and double, 2026-09-28",
    "covar_pop": "rejected on compute 2026-09-28",
    "map": "MAP( is a syntax error on compute, 2026-09-28",
}

# The same facts on the RENDERED side, plus forms that exist but are wrong here.
# Every SQL string the dialect tests compile is checked against these, so a
# fix made at one emission site cannot leave a sibling path emitting the form.
ABSENT_OR_UNSAFE_SQL: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in (
        (r"\bMAKE_DATE\s*\(", "absent"),
        (r"\bDATE_FROM_PARTS\s*\(", "absent"),
        (r"\bMAKE_TIME\s*\(", "absent"),
        (r"\bMAKE_TIMESTAMP\s*\(", "absent"),
        (r"\bDATE_BIN\s*\(", "absent"),
        (r"\bBIT_XOR\s*\(", "absent"),
        (r"\bARRAY_AGG\s*\(", "absent"),
        (r"\bFIRST\s*\(", "absent"),
        (r"\bMODE\s*\(", "absent"),
        (r"\bPERCENTILE_DISC\s*\(", "absent"),
        (r"\bREGEXP_MATCH\s*\(", "absent"),
        (r"\bHASHTEXTEXTENDED\s*\(", "absent"),
        (r"\bGEN_RANDOM_UUID\s*\(", "absent"),
        (r"\bENCODE\s*\(", "absent"),
        (r"\bPG_TYPEOF\s*\(", "absent"),
        (r"\bARRAY_REMOVE\s*\(", "absent"),
        (r"\bCARDINALITY\s*\(", "absent"),
        (r"\bCORR\s*\(", "absent"),
        (r"\bCOVAR_(POP|SAMP)\s*\(", "absent"),
        (r"\bMAP\s*\(", "absent"),
        (r"\bTO_TIMESTAMP\s*\([^,()]*\)", "one-argument form absent"),
        (
            r"\bREGEXP_REPLACE\s*\((?:[^()]|\([^()]*\))*,\s*'g'\s*\)",
            "4th argument is a position",
        ),
        (r"\bSTARTS_WITH\s*\(", "absent"),
        (r"\bFILTER\s*\(\s*WHERE\b", "no aggregate FILTER clause (2026-09-24)"),
        (r"'escape'\s*\)", "bytea escape format; DECODE is CASE-style here"),
        (
            r"\bANY_VALUE\s*\(",
            "exists, but may return NULL amid non-NULLs (AWS docs); NULL-unsafe",
        ),
    )
)


def absent_or_unsafe_forms(sql: str) -> list[str]:
    """Each denied form ``sql`` contains, with its reason; empty when clean."""
    return [
        f"{pattern.pattern} ({reason})"
        for pattern, reason in ABSENT_OR_UNSAFE_SQL
        if pattern.search(sql)
    ]
