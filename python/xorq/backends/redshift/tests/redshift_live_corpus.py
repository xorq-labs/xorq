"""What the live probes ask, kept apart from the tests that assert the answers.

``INHERITED`` holds one expression per function name the op inventory lists as
reachable through inherited code, built over the probe table so that every
compiled statement runs on compute; ``RENAMED`` does the same for the names
the xorq Redshift dialect renames typed nodes to. ``RECHECK`` holds the acceptances an
earlier record measured with constant-only queries, which ran on the leader
node, re-asked against the probe table. ``{table}`` is filled in with the
qualified probe table.
"""

from __future__ import annotations

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.types as ir


def _zero_string(t):
    """An empty string that depends on a column, so nothing folds on the leader."""
    return ibis.ifelse(t.id < 0, t.title, "")


def _ts(t):
    return (ibis.literal("2020-01-02 03:04:05.123456") + _zero_string(t)).cast(
        "timestamp"
    )


def _map(t):
    return ibis.map(ibis.array(["a"]), ibis.array([t.id]))


INHERITED = {
    # name: (the op that reaches it, expression factory)
    "avg": ("Mean", lambda t: t.fee_rate.mean()),
    "bit_and": ("BitAnd", lambda t: t.id.bit_and()),
    "bit_or": ("BitOr", lambda t: t.id.bit_or()),
    "bool_and": ("Min", lambda t: t.is_live.min()),
    "min": ("Min", lambda t: t.id.min()),
    "sum": ("Sum", lambda t: t.id.sum()),
    "corr": ("Correlation", lambda t: t.id.corr(t.id * 2, how="pop")),
    "ceil": ("Ceil", lambda t: t.fee_rate.ceil()),
    "floor": ("Floor", lambda t: t.fee_rate.floor()),
    "round": ("Round", lambda t: t.fee_rate.round(2)),
    "round/float": ("Round", lambda t: (t.fee_rate.cast("float64") + 0.5).round(2)),
    "round/float-large": (
        "Round",
        lambda t: (t.fee_rate.cast("float64") * 1e25).round(2),
    ),
    "modulus/float": ("Modulus", lambda t: (t.fee_rate.cast("float64") + 7) % 0.5),
    "modulus/decimal": ("Modulus", lambda t: (t.fee_rate + 7) % 2),
    "exp": ("E", lambda t: t.fee_rate + ibis.e),
    "log": ("Log", lambda t: (t.fee_rate + 1).log(3)),
    "log/log2": ("Log2", lambda t: (t.fee_rate + 1).log2()),
    "ln": ("Ln", lambda t: (t.fee_rate + 1).ln()),
    "ln/log10": ("Log10", lambda t: (t.fee_rate + 1).log10()),
    "sign": ("Sign", lambda t: t.id.sign()),
    "greatest": ("Greatest", lambda t: ibis.greatest(t.id, 1)),
    "least": ("Least", lambda t: ibis.least(t.id, 1)),
    "coalesce": ("Coalesce", lambda t: ibis.coalesce(t.title, "x")),
    "nullif": ("NullIf", lambda t: t.id.nullif(1)),
    "concat_ws": ("StringJoin", lambda t: ibis.literal(",").join([t.title, t.title])),
    "ltrim": ("LStrip", lambda t: t.title.lstrip()),
    "rtrim": ("RStrip", lambda t: t.title.rstrip()),
    "trim": ("Strip", lambda t: t.title.strip()),
    "strpos": ("StringContains", lambda t: t.title.contains("a")),
    "substr": ("StringFind", lambda t: t.title.find("a")),
    "substring": ("Substring", lambda t: t.title.substr(1, 2)),
    "regexp_like": ("RegexSearch", lambda t: t.title.re_search("a.")),
    "regexp_replace": ("RegexReplace", lambda t: t.title.re_replace("a", "b")),
    "rand": ("RandomScalar", lambda t: ibis.random() + t.id * 0),
    "pg_typeof": ("TypeOf", lambda t: t.id.typeof()),
    "date_trunc": ("TimestampTruncate", lambda t: _ts(t).truncate("D")),
    "extract": ("ExtractDay", lambda t: _ts(t).day()),
    "extract/epoch": ("ExtractEpochSeconds", lambda t: _ts(t).epoch_seconds()),
    "extract/millisecond": ("ExtractMillisecond", lambda t: _ts(t).millisecond()),
    "extract/week": ("ExtractWeekOfYear", lambda t: _ts(t).week_of_year()),
    "extract/dow": ("DayOfWeekIndex", lambda t: _ts(t).day_of_week.index()),
    "to_char": ("DayOfWeekName", lambda t: _ts(t).day_of_week.full_name()),
    "to_timestamp": ("Cast", lambda t: (t.id + 1_600_000_000).cast("timestamp")),
    "timezone": (
        "Cast",
        lambda t: (t.id + 1_600_000_000).cast(ibis.dtype("timestamp('UTC')")),
    ),
    "extract/cast": ("Cast", lambda t: _ts(t).cast("int64")),
    "exists": (
        "ExistsSubquery",
        lambda t: t.filter((t.view().id == t.id).any()).select("id").order_by("id"),
    ),
    "array": ("Array", lambda t: ibis.array([t.id, t.id])),
    "array_position": ("FindInSet", lambda t: t.title.find_in_set(["a", "b"])),
    "array_remove": ("ArrayRemove", lambda t: ibis.array([t.id, 1]).remove(1)),
    "cardinality": ("ArrayRepeat", lambda t: ibis.array([t.id]).repeat(2)),
    "generate_series": (
        "TimestampRange",
        lambda t: ibis.range(
            _ts(t), _ts(t) + ibis.interval(days=2), ibis.interval(days=1)
        ),
    ),
    "map": ("Map", lambda t: _map(t)),
    "akeys": ("MapKeys", lambda t: _map(t).keys()),
    "avals": ("MapValues", lambda t: _map(t).values()),
    "exist": ("MapContains", lambda t: _map(t).contains("a")),
    "jsonb_extract_path_text": ("MapGet", lambda t: _map(t).get("a")),
    "row": ("StructColumn", lambda t: ibis.struct({"a": t.id})),
    "jsonb_extract_path": (
        "JSONGetItem",
        lambda t: (ibis.literal('{"a": 1}') + _zero_string(t)).cast("json")["a"],
    ),
}

# One expression per function name the xorq Redshift dialect's own TRANSFORMS
# rename a typed sqlglot node to. The op inventory's static read cannot see
# these, because no visitor names them; see ``DIALECT_RENAMED`` there.
RENAMED = {
    "split_to_array": ("StringSplit", lambda t: t.title.split(" ")),
    "get_array_length": ("ArrayLength", lambda t: ibis.array([t.id, t.id]).length()),
    "get_array_length/split": (
        "ArrayLength",
        lambda t: t.title.split(" ").length(),
    ),
}

# The kinds of expression the corpus builds. A reduction is compiled as an
# aggregate, everything else as a projection alongside the id column.
REDUCTIONS = frozenset({"avg", "bit_and", "bit_or", "bool_and", "min", "sum", "corr"})


def compile_inherited(name, table, compiler, corpus=INHERITED):
    """The statement ``corpus[name]`` compiles to over ``table``."""
    _, build = corpus[name]
    value = build(table)
    if isinstance(value, ir.Table):
        expr = value
    elif name in REDUCTIONS:
        expr = table.aggregate(v=value)
    else:
        expr = table.select(table.id, v=value).order_by("id").limit(2)
    return compiler.to_sqlglot(expr).sql(dialect=compiler.dialect)


RECHECK = {
    # name: SQL, one question each, the earlier leader-node answer in the key
    "tab_is_decoded": (
        "SELECT LENGTH(LEFT(title, 0) || '\\t'), "
        "CASE WHEN LEFT(title, 0) || '\\t' = CHR(9 + (id - id)::int) "
        "THEN 'DECODED' ELSE 'LITERAL' END FROM {table} LIMIT 1"
    ),
    "regex_doubled_backslash_is_a_digit_class": (
        "SELECT SUM(CASE WHEN id::varchar ~ '\\\\d' THEN 1 ELSE 0 END), "
        "COUNT(*) FROM {table}"
    ),
    "regex_single_backslash_d_is_the_letter": (
        "SELECT SUM(CASE WHEN id::varchar ~ '\\d' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN LEFT(title, 0) || 'd' ~ '\\d' THEN 1 ELSE 0 END), "
        "COUNT(*) FROM {table}"
    ),
    "lag_takes_no_frame": "SELECT LAG(id) OVER (ORDER BY id) FROM {table} LIMIT 1",
    "lead_takes_no_frame": "SELECT LEAD(id) OVER (ORDER BY id) FROM {table} LIMIT 1",
    "median_over_partition": (
        "SELECT MEDIAN(fee_rate) OVER (PARTITION BY is_live) FROM {table} LIMIT 1"
    ),
    "percentile_cont_over_partition": (
        "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fee_rate) "
        "OVER (PARTITION BY is_live) FROM {table} LIMIT 1"
    ),
    "listagg_over_partition": (
        "SELECT LISTAGG(title, ',') OVER (PARTITION BY is_live) FROM {table} LIMIT 1"
    ),
    "row_number_takes_no_frame": (
        "SELECT ROW_NUMBER() OVER (ORDER BY id) FROM {table} LIMIT 1"
    ),
    "count_case": ("SELECT COUNT(CASE WHEN is_live THEN 1 ELSE NULL END) FROM {table}"),
    "count_distinct_case": (
        "SELECT COUNT(DISTINCT CASE WHEN is_live THEN title ELSE NULL END) FROM {table}"
    ),
    "listagg_case_constant_delimiter": (
        "SELECT LISTAGG(CASE WHEN is_live THEN title ELSE NULL END, ',') FROM {table}"
    ),
    "listagg_within_group": (
        "SELECT LISTAGG(title, ',') WITHIN GROUP (ORDER BY id) FROM {table}"
    ),
    "cast_timestamptz": (
        "SELECT CAST('2026-09-24 00:00:00+00' || LEFT(title, 0) AS TIMESTAMPTZ) "
        "FROM {table} LIMIT 1"
    ),
}
