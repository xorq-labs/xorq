"""The dialect-delta inventory: what ``dialect = Redshift`` actually changed.

``RedshiftCompiler.dialect = Redshift`` is one token, and it retargets three
layers at once -- generator, parser and tokenizer. The commit that wrote it
measured its effect on the four constructs then under investigation, found
three of them byte-identical, and concluded the line was close to inert. That
conclusion was correct about those four and wrong about the line: the two
dialects differ in **67** places, and a later review found defects in three of
the 63 nobody had looked at (string-escape rules, ``StartsWith``, ``Pow``).

So this file exists to make the blast radius of that one token countable. It
computes the live difference set between the dialect the Postgres compiler uses
and the dialect the Redshift compiler uses, and asserts it equals the inventory
below. Every entry carries a tag:

``ACCEPTED``
    The difference is real, desirable or harmless, and the emitted SQL is
    better for it. Most entries: they are why the retarget was done.
``OVERRIDDEN``
    sqlglot's answer is not used -- ``RedshiftCompiler`` or xorq's ``Redshift``
    dialect subclass decides this above sqlglot. The note names where.
``KNOWN_OPEN``
    A difference whose correctness cannot be settled offline. Cross-referenced
    to the OPEN WAREHOUSE QUESTIONS block in ``test_redshift_dialect.py``.

The test fails when the live set and the inventory disagree, in either
direction. That is the point: a sqlglot upgrade that adds, drops or rewrites a
Redshift transform cannot land silently -- someone has to look at it and write
down which of the three tags it gets. Re-classification on every bump is the
feature, not the maintenance cost, because a bump is exactly when these change
underneath the backend.

Sited in ``python/xorq/tests/`` rather than under the backend directory for the
reason ``test_redshift_backend.py`` gives: ``backends/conftest.py`` auto-applies
``pytest.mark.<backend>`` by path and every CI job selects by marker, so a test
placed there would run only in the credential-gated workflow. Nothing here
needs a warehouse -- it is pure class introspection.
"""

from __future__ import annotations

import pytest

from xorq.backends.postgres.compiler import PostgresCompiler
from xorq.backends.redshift.compiler import RedshiftCompiler


ACCEPTED = "accepted"
OVERRIDDEN = "overridden"
KNOWN_OPEN = "known-open"

INVENTORY: dict[str, tuple[str, str]] = {
    # -- Tokenizer -----------------------------------------------------------
    "Tokenizer.STRING_ESCAPES": (
        ACCEPTED,
        "VERIFIED 2026-09-24: LENGTH('\\t') is 1 on the warehouse, so Redshift "
        "decodes backslash escapes and the rs spelling is the correct one. "
        "This also makes the change a silent BUG FIX for the four regex ops: "
        "under the pg dialect '1' ~ '\\d' did not match while 'd' ~ '\\d' did, "
        "i.e. every backslash class was being sent as a literal letter.",
    ),
    "Tokenizer.BIT_STRINGS": (ACCEPTED, "Redshift has no b'' literal syntax."),
    "Tokenizer.HEX_STRINGS": (ACCEPTED, "Redshift has no x'' literal syntax."),
    # -- Parser --------------------------------------------------------------
    "Parser.SUPPORTS_IMPLICIT_UNNEST": (
        ACCEPTED,
        "Parse-side only; this backend never parses Redshift SQL. Note the "
        "generator side is the opposite story -- see TRANSFORMS.differs.Explode.",
    ),
    # -- Generator scalars ---------------------------------------------------
    "Generator.MULTI_ARG_DISTINCT": (
        OVERRIDDEN,
        "sqlglot says rs True, meaning COUNT(DISTINCT a, b) is available. "
        "VERIFIED 2026-09-24 that it is NOT: the warehouse rejects both that "
        "form ('function count(bigint, varchar) does not exist') and the "
        "postgres row-constructor form ('could not identify an equality "
        "operator for type record'). `visit_CountDistinctStar` raises rather "
        "than trusting either. The one place sqlglot's Redshift model is "
        "measurably wrong about the warehouse.",
    ),
    "Generator.SUPPORTS_MEDIAN": (
        ACCEPTED,
        "rs True, and VERIFIED: MEDIAN(col) runs. `visit_Quantile` lowers to "
        "PERCENTILE_CONT regardless -- the more general spelling, and the one "
        "whose WITHIN GROUP carries the filter predicate. Both were verified "
        "2026-09-24 to run and to respect the folded predicate (0.02750 "
        "filtered vs 0.02875 unfiltered over xorq_test.offers). Note "
        "PERCENTILE_DISC, the non-numeric branch, is unsupported outright -- "
        "visit_Quantile raises there.",
    ),
    "Generator.ALTER_SET_TYPE": (ACCEPTED, "DDL only; backend emits no ALTER."),
    "Generator.CAN_IMPLEMENT_ARRAY_ANY": (
        ACCEPTED,
        "rs False -- consistent with Redshift having no array type. The array "
        "ops are in the compiler's UNSUPPORTED_OPS.",
    ),
    "Generator.COPY_PARAMS_ARE_WRAPPED": (ACCEPTED, "COPY only; ingest uses ADBC."),
    "Generator.EXCEPT_INTERSECT_SUPPORT_ALL_CLAUSE": (
        ACCEPTED,
        "rs False: no EXCEPT ALL / INTERSECT ALL. A real Redshift limit.",
    ),
    "Generator.HEX_FUNC": (ACCEPTED, "TO_HEX is the Redshift spelling."),
    "Generator.LAST_DAY_SUPPORTS_DATE_PART": (ACCEPTED, "rs False. Real limit."),
    "Generator.LIMIT_FETCH": (ACCEPTED, "Redshift has no FETCH FIRST."),
    "Generator.LOCKING_READS_SUPPORTED": (
        ACCEPTED,
        "rs False: no SELECT ... FOR UPDATE. Not reachable from this backend.",
    ),
    "Generator.NVL2_SUPPORTED": (ACCEPTED, "rs True. Redshift has NVL2."),
    "Generator.PARSE_JSON_NAME": (ACCEPTED, "JSON_PARSE is the Redshift spelling."),
    "Generator.SUPPORTS_BETWEEN_FLAGS": (ACCEPTED, "rs False. Real limit."),
    "Generator.SUPPORTS_CONVERT_TIMEZONE": (ACCEPTED, "rs True. Real function."),
    "Generator.SUPPORTS_DECODE_CASE": (ACCEPTED, "rs True. Redshift has DECODE."),
    "Generator.TZ_TO_WITH_TIME_ZONE": (ACCEPTED, "Spelling of a tz cast."),
    "Generator.VALUES_AS_TABLE": (
        ACCEPTED,
        "rs False: VALUES is not a standalone table on Redshift. Affects "
        "memtable rendering; the backend ingests through ADBC instead.",
    ),
    "Generator.WITH_PROPERTIES_PREFIX": (ACCEPTED, "DDL properties only."),
    # -- TRANSFORMS present for Postgres and not for Redshift ----------------
    "TRANSFORMS.pg-only.DateFromParts": (
        OVERRIDDEN,
        "Neither MAKE_DATE nor DATE_FROM_PARTS exists on Redshift. Lowered in "
        "RedshiftCompiler.visit_DateFromYMD to TO_DATE(LPAD(...)||...).",
    ),
    "TRANSFORMS.pg-only.RegexpSplit": (
        OVERRIDDEN,
        "Redshift has no regex-split-to-array under any name, so declining the "
        "postgres rename only moves it to REGEXP_SPLIT, which no engine has. "
        "ops.RegexSplit is in the compiler's UNSUPPORTED_OPS.",
    ),
    "TRANSFORMS.pg-only.AnyValue": (
        ACCEPTED,
        "Postgres renames AnyValue away because it lacked it before 16; "
        "Redshift has ANY_VALUE natively, which is what SIMPLE_OPS now targets "
        "for ops.Arbitrary.",
    ),
    "TRANSFORMS.pg-only.Pow": (
        ACCEPTED,
        "sqlglot's Redshift already renders POWER(...). An override here was "
        "byte-identical to none and was removed as dead code.",
    ),
    "TRANSFORMS.pg-only.Getbit": (ACCEPTED, "Not reachable from any ibis op."),
    "TRANSFORMS.pg-only.LastDay": (ACCEPTED, "Redshift has LAST_DAY natively."),
    "TRANSFORMS.pg-only.ParseJSON": (ACCEPTED, "Covered by PARSE_JSON_NAME."),
    "TRANSFORMS.pg-only.Pivot": (ACCEPTED, "Not emitted by this compiler."),
    "TRANSFORMS.pg-only.Round": (ACCEPTED, "Redshift ROUND needs no rewrite."),
    "TRANSFORMS.pg-only.SHA2": (ACCEPTED, "Redshift has SHA2 natively."),
    # -- TRANSFORMS present for Redshift and not for Postgres ----------------
    "TRANSFORMS.rs-only.StartsWith": (
        OVERRIDDEN,
        "sqlglot lowers STARTS_WITH to `s LIKE p || '%'` with p interpolated "
        "raw, so a % or _ in the operand becomes a wildcard -- silent wrong "
        "rows. RedshiftCompiler.visit_StartsWith emits LEFT(s, LENGTH(p)) = p "
        "instead, the shape visit_EndsWith already had.",
    ),
    "TRANSFORMS.rs-only.StringToArray": (
        OVERRIDDEN,
        "xorq's Redshift subclass pins Split to SPLIT_TO_ARRAY; see "
        "dialects.py. Tested by test_array_ops_with_a_redshift_spelling_are_kept.",
    ),
    "TRANSFORMS.rs-only.ApproxDistinct": (
        ACCEPTED,
        "Redshift's APPROXIMATE COUNT(DISTINCT ...). Reached only through "
        "visit_ApproxCountDistinct, which this backend aliases to the exact "
        "count to keep DISTINCT out of the CASE fallback.",
    ),
    "TRANSFORMS.rs-only.Concat": (
        ACCEPTED,
        "Lowers concat to the || dpipe form, which is what visit_DateFromYMD "
        "relies on for its ISO-8601 assembly.",
    ),
    "TRANSFORMS.rs-only.ConcatWs": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.rs-only.FarmFingerprint": (ACCEPTED, "Not emitted by this compiler."),
    "TRANSFORMS.rs-only.FromBase": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.rs-only.Hex": (ACCEPTED, "Covered by HEX_FUNC."),
    "TRANSFORMS.rs-only.RegexpExtract": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.rs-only.TableSample": (ACCEPTED, "Not emitted by this compiler."),
    "TRANSFORMS.rs-only.DistKeyProperty": (ACCEPTED, "Redshift-only DDL property."),
    "TRANSFORMS.rs-only.DistStyleProperty": (ACCEPTED, "Redshift-only DDL property."),
    "TRANSFORMS.rs-only.SortKeyProperty": (ACCEPTED, "Redshift-only DDL property."),
    "TRANSFORMS.rs-only.GeneratedAsIdentityColumnConstraint": (
        ACCEPTED,
        "DDL only; ingest does not emit identity columns.",
    ),
    # -- TRANSFORMS present in both, implemented differently -----------------
    "TRANSFORMS.differs.Explode": (
        OVERRIDDEN,
        "THE ONE THAT BITES. sqlglot's Redshift generator does not raise on "
        "UNNEST/EXPLODE -- it warns and returns the empty string, which flows "
        "back into expression building as a str. Result: an internal "
        "AttributeError from inside sqlglot, or SQL with a hole in it. The "
        "twelve unnest-dependent array ops are in the compiler's "
        "UNSUPPORTED_OPS so they raise OperationNotDefinedError instead.",
    ),
    "TRANSFORMS.differs.ArraySize": (
        OVERRIDDEN,
        "xorq's Redshift subclass pins this to GET_ARRAY_LENGTH.",
    ),
    "TRANSFORMS.differs.Split": (
        OVERRIDDEN,
        "xorq's Redshift subclass pins this to SPLIT_TO_ARRAY.",
    ),
    "TRANSFORMS.differs.GroupConcat": (
        OVERRIDDEN,
        "Renders LISTAGG. visit_GroupConcat controls the argument shape above "
        "it: predicate on the value only (the delimiter must stay constant) "
        "and ordering as WITHIN GROUP, not inside the argument list.",
    ),
    "TRANSFORMS.differs.ArrayConcat": (ACCEPTED, "Redshift ARRAY_CONCAT spelling."),
    "TRANSFORMS.differs.CurrentTimestamp": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.differs.DateAdd": (ACCEPTED, "DATEADD, Redshift's spelling."),
    "TRANSFORMS.differs.DateDiff": (ACCEPTED, "DATEDIFF, Redshift's spelling."),
    "TRANSFORMS.differs.JSONExtract": (ACCEPTED, "Redshift JSON spelling."),
    "TRANSFORMS.differs.JSONExtractScalar": (ACCEPTED, "Redshift JSON spelling."),
    "TRANSFORMS.differs.SHA2Digest": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.differs.TsOrDsAdd": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.differs.TsOrDsDiff": (ACCEPTED, "Redshift spelling."),
    "TRANSFORMS.differs.UnixToTime": (ACCEPTED, "Redshift spelling."),
    # -- TYPE_MAPPING --------------------------------------------------------
    "TYPE_MAPPING.BINARY": (ACCEPTED, "VARBYTE, not BYTEA. A real fix."),
    "TYPE_MAPPING.BLOB": (ACCEPTED, "VARBYTE, not BYTEA."),
    "TYPE_MAPPING.VARBINARY": (ACCEPTED, "VARBYTE, not BYTEA."),
    "TYPE_MAPPING.ROWVERSION": (ACCEPTED, "VARBYTE, not BYTEA."),
    "TYPE_MAPPING.INT": (ACCEPTED, "INTEGER, Redshift's spelling."),
    "TYPE_MAPPING.TIMESTAMPTZ": (
        ACCEPTED,
        "rs renders TIMESTAMPTZ as bare TIMESTAMP where RedshiftType (the ibis "
        "type mapper, a different layer) renders TIMESTAMP WITH TIME ZONE. "
        "VERIFIED 2026-09-24 that Redshift accepts BOTH spellings and they "
        "round-trip identically, so the two layers disagreeing costs nothing "
        "today. Worth revisiting only if a CREATE starts going through the "
        "sqlglot layer for a tz-bearing column.",
    ),
    "TYPE_MAPPING.TIMETZ": (ACCEPTED, "Same as TIMESTAMPTZ; both accepted."),
}


def _scalar_attrs(cls):
    out = {}
    for name in dir(cls):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(cls, name, None)
        if callable(value) or isinstance(value, (dict, set, frozenset)):
            continue
        try:
            hash(value)
        except TypeError:
            value = repr(value)
        out[name] = value
    return out


def _fingerprint(func):
    """Identify a TRANSFORMS entry by behaviour, not object identity.

    ``rename_func("power")`` builds a fresh closure on every call, so two
    identical entries are never the same object. The closed-over strings are
    what actually distinguish ``rename_func("power")`` from
    ``rename_func("pow")``, so they are the fingerprint.
    """
    cells = tuple(
        cell.cell_contents
        for cell in (getattr(func, "__closure__", None) or ())
        if isinstance(cell.cell_contents, str)
    )
    return (getattr(func, "__qualname__", repr(func)), cells)


def live_delta():
    """Every way the Redshift dialect differs from the Postgres one."""
    pg = PostgresCompiler.dialect
    rs = RedshiftCompiler.dialect
    delta = set()

    for layer in ("Generator", "Parser", "Tokenizer"):
        left = _scalar_attrs(getattr(pg, layer))
        right = _scalar_attrs(getattr(rs, layer))
        for name in set(left) | set(right):
            sentinel = "<absent>"
            if left.get(name, sentinel) != right.get(name, sentinel):
                delta.add(f"{layer}.{name}")

    left, right = pg.Generator.TRANSFORMS, rs.Generator.TRANSFORMS
    left_names = {k.__name__ for k in left}
    right_names = {k.__name__ for k in right}
    delta |= {f"TRANSFORMS.pg-only.{n}" for n in left_names - right_names}
    delta |= {f"TRANSFORMS.rs-only.{n}" for n in right_names - left_names}
    delta |= {
        f"TRANSFORMS.differs.{k.__name__}"
        for k in left
        if k in right and _fingerprint(left[k]) != _fingerprint(right[k])
    }

    left, right = pg.Generator.TYPE_MAPPING, rs.Generator.TYPE_MAPPING
    delta |= {
        f"TYPE_MAPPING.{k.name}"
        for k in set(left) | set(right)
        if left.get(k) != right.get(k)
    }

    return delta


def test_the_dialect_delta_matches_the_inventory():
    """Every difference is classified, and only classified differences exist.

    A sqlglot upgrade that adds, drops or rewrites a Redshift transform lands
    here as a failure naming the entry, which someone then tags. That is the
    whole mechanism: it converts an invisible change in a dependency into a
    required decision.
    """
    live = live_delta()
    recorded = set(INVENTORY)
    unclassified = sorted(live - recorded)
    stale = sorted(recorded - live)
    assert not unclassified, (
        "the Redshift dialect differs from Postgres in ways nobody has "
        "classified -- tag each as accepted / overridden / known-open in "
        f"INVENTORY: {unclassified}"
    )
    assert not stale, (
        "INVENTORY records differences that no longer exist -- sqlglot "
        f"probably changed underneath; re-check and remove: {stale}"
    )


def test_the_delta_is_large_enough_to_be_worth_inventorying():
    """Guards the computation itself.

    If ``live_delta`` silently degenerates -- an attribute rename upstream, a
    layer that stops existing -- the test above passes by comparing an empty
    set to an empty set. This is the tripwire for that.
    """
    assert len(live_delta()) > 50


@pytest.mark.parametrize("tag", [ACCEPTED, OVERRIDDEN])
def test_every_tag_is_in_use(tag):
    """Each classification that should have members has them.

    ``KNOWN_OPEN`` is deliberately excluded: as of the 2026-09-24 warehouse
    session it has none, which is the desired state and is asserted directly by
    ``test_known_open_entries_are_the_documented_ones``.
    """
    assert any(entry[0] == tag for entry in INVENTORY.values())


def test_known_open_entries_are_the_documented_ones():
    """The KNOWN_OPEN set is small, and every member is a live question.

    Pinned so that tagging something known-open is a deliberate act rather than
    a way to make this file stop complaining.
    """
    known_open = {k for k, (tag, _) in INVENTORY.items() if tag == KNOWN_OPEN}
    assert known_open == set(), (
        "every dialect difference was settled against the xorq-test warehouse "
        f"on 2026-09-24; these are newly open and need an answer: {known_open}"
    )
