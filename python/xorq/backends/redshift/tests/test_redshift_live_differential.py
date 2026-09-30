"""Differential execution: the same expressions on Redshift and on PostgreSQL,
over the same rows, compared by value.

String tests restate what their author expected, so they cannot see a lowering
that runs and answers wrongly. Eight such defects reached this backend: a
rolling median widened to the partition, lenient TO_DATE, truncating LPAD,
blank-insensitive startswith, a binary literal storing its escape text,
NULL-unsafe ANY_VALUE, an unordered windowed listagg, and an off-by-one
find_in_set. Each shows up here as a value mismatch against PostgreSQL.

The rows are read from the Redshift probe table into a TEMPORARY table in the
local PostgreSQL (``POSTGRES_*``); nothing is written to Redshift. A difference
that is intended is listed in ``INTENDED_DIFFERENCES`` with its reason. An
unlisted difference fails, and so does a listed one that stops differing.
"""

from __future__ import annotations

import datetime
import decimal
import math

import pytest

import xorq.vendor.ibis as ibis
from xorq.backends.postgres.compiler import compiler as postgres_compiler
from xorq.backends.redshift.compiler import compiler as redshift_compiler
from xorq.backends.redshift.tests.redshift_live_corpus import (
    INHERITED,
    compile_inherited,
)
from xorq.backends.redshift.tests.redshift_live_harness import PROBE_TABLE_SCHEMA


psycopg = pytest.importorskip("psycopg")
postgres_utils = pytest.importorskip("xorq.common.utils.postgres_utils")

_LOCAL_TABLE = "redshift_probe_rows"


def _t(t):
    return ibis.ifelse(t.id < 0, t.title, "")


# Expressions aimed at the defects above, in their fixed forms. Each builds a
# whole table expression over the probe table.
TARGETED = {
    "startswith_trailing_blank": lambda t: t.aggregate(
        n=ibis.ifelse(t.title.startswith(t.title + " "), 1, 0).sum()
    ),
    "endswith_trailing_blank": lambda t: t.aggregate(
        n=ibis.ifelse(t.title.endswith(t.title + " "), 1, 0).sum()
    ),
    "find_in_set": lambda t: t.select(
        "id", v=t.title.find_in_set(["Bravo Rewards", "Alpha Cashback", "nope"])
    ).order_by("id"),
    "binary_literal": lambda t: (
        t.select("id", v=ibis.literal(b"ab").cast("string")).order_by("id").limit(1)
    ),
    "date_from_parts": lambda t: t.select(
        "id", v=ibis.date(t.id * 0 + 2026, t.id * 0 + 2, t.id % 28 + 1)
    ).order_by("id"),
    "windowed_listagg_order": lambda t: (
        t.select("id", v=t.title.group_concat(",").over(ibis.window(order_by=t.id)))
        .order_by("id")
        .limit(1)
    ),
    "median_over_partition": lambda t: t.select(
        "id", v=t.fee_rate.median().over(ibis.window(group_by=t.is_live))
    ).order_by("id"),
    "arbitrary_where_is_not_null": lambda t: t.aggregate(
        v=t.title.arbitrary(where=t.id == 3).notnull()
    ),
    "epoch_to_timestamp": lambda t: t.select(
        "id", v=(t.id + 1_600_000_000).cast("timestamp")
    ).order_by("id"),
    "regex_replace_all": lambda t: t.select(
        "id", v=t.title.re_replace("a", "_")
    ).order_by("id"),
    "log_base": lambda t: t.select("id", v=(t.fee_rate + 1).log(3)).order_by("id"),
    "split_length": lambda t: t.select("id", v=t.title.split(" ").length()).order_by(
        "id"
    ),
    "split_length_empty": lambda t: t.select(
        "id", v=_t(t).split(" ").length()
    ).order_by("id"),
    "array_length": lambda t: t.select(
        "id", v=ibis.array([t.id, t.id, t.id]).length()
    ).order_by("id"),
}

# Nondeterministic, so not comparable by value.
_NONDETERMINISTIC = {"rand"}

INTENDED_DIFFERENCES = {
    "array": "Redshift builds a SUPER, which arrives as JSON text",
    "jsonb_extract_path": "Redshift returns text where PostgreSQL returns jsonb",
    "avg": (
        "ibis types the mean of decimal(18, 4) as decimal(18, 4); Redshift "
        "truncates to that scale while PostgreSQL carries every digit"
    ),
    # The same declared type, for logarithms: both compilers cast log(b) and
    # log2 to decimal(18, 4), but PostgreSQL's ln and log10 skip the cast and
    # carry every digit. Redshift casts all four.
    "ln": "ibis types ln of decimal(18, 4) as decimal(18, 4); see avg",
    "ln/log10": "ibis types log10 of decimal(18, 4) as decimal(18, 4); see avg",
    "extract/epoch": (
        "Redshift's EXTRACT(EPOCH) is whole seconds, ibis's integer type; "
        "PostgreSQL returns the fraction"
    ),
    "extract/cast": "the same EXTRACT(EPOCH), reached through a cast",
}

# Cases only PostgreSQL fails, so there is nothing to compare: the test holds
# that Redshift runs them. When PostgreSQL starts to run one, the entry fails
# and must be removed, and the pair becomes comparable.
POSTGRES_CANNOT = {
    "median_over_partition": "PostgreSQL has no ordered-set aggregate over a window",
    "regexp_replace": (
        "xorq's PostgreSQL backend emits REGEXP_REPLACE(..., 'g', 'g'), which "
        "PostgreSQL rejects"
    ),
    "regex_replace_all": "the same PostgreSQL REGEXP_REPLACE defect",
    "binary_literal": (
        "xorq's PostgreSQL backend spells b'ab' as '\\x61\\x62', which is "
        "not bytea hex input"
    ),
}


def _normalise(value):
    if isinstance(value, decimal.Decimal | float):
        return round(float(value), 6)
    if isinstance(value, bytes | memoryview):
        return bytes(value).hex()
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None)
    return value


def _rows_match(left, right):
    if len(left) != len(right):
        return False
    for lrow, rrow in zip(left, right):
        for lval, rval in zip(lrow, rrow):
            lnorm, rnorm = _normalise(lval), _normalise(rval)
            if isinstance(lnorm, float) and isinstance(rnorm, float):
                if not math.isclose(lnorm, rnorm, rel_tol=1e-4, abs_tol=1e-6):
                    return False
            elif lnorm != rnorm:
                return False
    return True


@pytest.fixture(scope="module")
def local_postgres(live_config, run_select):
    config = postgres_utils.postgres_config
    if not config["POSTGRES_HOST"]:
        pytest.skip("differential execution needs the local POSTGRES_* database")
    con = psycopg.connect(
        host=config["POSTGRES_HOST"],
        port=int(config["POSTGRES_PORT"]),
        dbname=config["POSTGRES_DATABASE"],
        user=config["POSTGRES_USER"],
        password=config["POSTGRES_PASSWORD"],
        autocommit=True,
    )
    rows = run_select(
        "SELECT id, title, is_live, fee_rate FROM "
        f"{live_config['XORQ_REDSHIFT_PROBE_TABLE']}"
    )
    con.execute(
        f"CREATE TEMPORARY TABLE {_LOCAL_TABLE} "
        "(id bigint, title varchar, is_live boolean, fee_rate numeric(18, 4))"
    )
    with con.cursor() as cur:
        cur.executemany(f"INSERT INTO {_LOCAL_TABLE} VALUES (%s, %s, %s, %s)", rows)
    yield con
    con.close()


def _both(build, probe_table, run_select, local_postgres):
    """(redshift rows or exception, postgres rows or exception)."""
    local_table = ibis.table(PROBE_TABLE_SCHEMA, name=_LOCAL_TABLE)
    results = []
    for compile_, run in (
        (lambda: build(probe_table, redshift_compiler), run_select),
        (
            lambda: build(local_table, postgres_compiler),
            lambda sql: local_postgres.execute(sql).fetchall(),
        ),
    ):
        try:
            results.append(run(compile_()))
        except Exception as e:  # noqa: BLE001 -- the error is the observation
            results.append(e)
    return results


def _assert_agree(name, redshift, postgres):
    if name in POSTGRES_CANNOT:
        assert isinstance(postgres, Exception), (
            f"{name}: PostgreSQL now runs this; drop it from POSTGRES_CANNOT"
        )
        assert not isinstance(redshift, Exception), f"{name}: {redshift!r}"
        return
    if isinstance(redshift, Exception) or isinstance(postgres, Exception):
        both_failed = isinstance(redshift, Exception) and isinstance(
            postgres, Exception
        )
        refused = type(redshift).__module__.startswith("xorq")
        assert both_failed or refused, (
            f"{name}: only one engine failed -- redshift={redshift!r} "
            f"postgres={postgres!r}"
        )
        return
    agree = _rows_match(redshift, postgres)
    if name in INTENDED_DIFFERENCES:
        assert not agree, f"{name} no longer differs; drop it from INTENDED_DIFFERENCES"
    else:
        assert agree, f"{name}: redshift={redshift[:3]!r} postgres={postgres[:3]!r}"


@pytest.mark.parametrize("name", sorted(set(INHERITED) - _NONDETERMINISTIC))
def test_inherited_expression_agrees_with_postgres(
    name, probe_table, run_select, local_postgres
):
    redshift, postgres = _both(
        lambda table, compiler: compile_inherited(name, table, compiler),
        probe_table,
        run_select,
        local_postgres,
    )
    _assert_agree(name, redshift, postgres)


@pytest.mark.parametrize("name", sorted(TARGETED))
def test_targeted_defect_agrees_with_postgres(
    name, probe_table, run_select, local_postgres
):
    redshift, postgres = _both(
        lambda table, compiler: compiler.to_sqlglot(TARGETED[name](table)).sql(
            dialect=compiler.dialect
        ),
        probe_table,
        run_select,
        local_postgres,
    )
    _assert_agree(name, redshift, postgres)
