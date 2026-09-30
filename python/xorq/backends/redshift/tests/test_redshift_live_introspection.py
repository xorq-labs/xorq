"""Both introspection paths, both read paths, over one table of every type.

``con.table`` types a column from its ``svv_all_columns`` spelling and
``con.sql`` from a result description's OID; a read is then served by ADBC or,
failing that, by psycopg. Each pairing has been wrong on its own: a
``VARBYTE`` value came back as its hex text on psycopg, a temporary table was
shadowed by the permanent one it was named after, and ``BIGINT`` bound as an
8-bit integer through ``con.sql``. Offline tests replay what was measured, so
they cannot see a type nobody measured. This reads a real column of each type
through every pairing and compares it with the value that was inserted.

It needs ``MATRIX_TABLE`` to exist in the probe table's schema, created by
``MATRIX_DDL`` and ``MATRIX_INSERT``; without it every test here skips. The
one write is the temporary table ``test_a_temporary_table_shadows_...``
creates, which ends with its own connection.
"""

from __future__ import annotations

import datetime
import decimal
from typing import Any

import pytest

import xorq.api as xo
import xorq.common.exceptions as exc
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.util as ibis_util
from xorq.backends.redshift import Backend as RedshiftBackend
from xorq.backends.redshift.tests.redshift_live_harness import RedshiftLiveConfig
from xorq.vendor.ibis.expr import types as ir


MATRIX_TABLE = "introspection_matrix"

ENTRIES = [pytest.param("table", id="table"), pytest.param("sql", id="sql")]

# (column, Redshift DDL type, dtype through the catalog path). The query path
# binds the same dtype, nullable, because a result description carries no
# nullability.
MATRIX_COLUMNS = (
    ("id", "BIGINT NOT NULL", dt.Int64(nullable=False)),
    ("i2", "SMALLINT", dt.Int16()),
    ("i4", "INTEGER", dt.Int32()),
    ("i8", "BIGINT", dt.Int64()),
    ("num", "NUMERIC(12,4)", dt.Decimal(12, 4)),
    ("f4", "REAL", dt.Float32()),
    ("f8", "DOUBLE PRECISION", dt.Float64()),
    ("flag", "BOOLEAN", dt.Boolean()),
    ("vc", "VARCHAR(32)", dt.String()),
    ("ch", "CHAR(4)", dt.String()),
    ("d", "DATE", dt.Date()),
    ("ts", "TIMESTAMP", dt.Timestamp(scale=6)),
    ("tstz", "TIMESTAMPTZ", dt.Timestamp(timezone="UTC", scale=6)),
    ("t", "TIME", dt.Time()),
    ("bin", "VARBYTE(16)", dt.Binary()),
    ("sup", "SUPER", dt.NamedUnknown(raw_type="super")),
)

UNMAPPABLE = ("sup",)

# Row 1 carries the values most likely to be mangled: a BIGINT past 2**53, a
# multibyte string, a binary value with a zero byte and a high byte, and
# microseconds. Row 2 carries the negatives and edges, a trailing blank among
# them. Row 3 is NULL wherever NULL is allowed.
MATRIX_ROWS = (
    (
        "1, 7, 70000, 9007199254740993, 12345678.1234, 1.5, 0.1, TRUE,"
        " 'héllo', 'ab', '2026-09-30', '2026-09-30 12:34:56.123456',"
        " '2026-09-30 12:34:56.123456+00', '12:34:56.123456',"
        " FROM_HEX('ab00ff'), JSON_PARSE('{\"a\": 1}')",
        {
            "id": 1,
            "i2": 7,
            "i4": 70000,
            "i8": 9007199254740993,
            "num": decimal.Decimal("12345678.1234"),
            "f4": 1.5,
            "f8": 0.1,
            "flag": True,
            "vc": "héllo",
            "ch": "ab  ",
            "d": datetime.date(2026, 9, 30),
            "ts": datetime.datetime(2026, 9, 30, 12, 34, 56, 123456),
            "tstz": datetime.datetime(
                2026, 9, 30, 12, 34, 56, 123456, tzinfo=datetime.timezone.utc
            ),
            "t": datetime.time(12, 34, 56, 123456),
            "bin": b"\xab\x00\xff",
        },
    ),
    (
        "2, -7, -1, -1, -0.0001, -2.25, -1e300, FALSE, 'trailing ', 'abcd',"
        " '1970-01-01', '1970-01-01 00:00:00', '1970-01-01 00:00:00+00',"
        " '00:00:00', FROM_HEX('00'), JSON_PARSE('[1, 2]')",
        {
            "id": 2,
            "i2": -7,
            "i4": -1,
            "i8": -1,
            "num": decimal.Decimal("-0.0001"),
            "f4": -2.25,
            "f8": -1e300,
            "flag": False,
            "vc": "trailing ",
            "ch": "abcd",
            "d": datetime.date(1970, 1, 1),
            "ts": datetime.datetime(1970, 1, 1),
            "tstz": datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc),
            "t": datetime.time(0, 0),
            "bin": b"\x00",
        },
    ),
    (
        "3" + ", NULL" * (len(MATRIX_COLUMNS) - 1),
        {"id": 3} | {name: None for name, *_ in MATRIX_COLUMNS[1:]},
    ),
)

EXPECTED_ROWS = [
    {k: v for k, v in expected.items() if k not in UNMAPPABLE}
    for _, expected in MATRIX_ROWS
]


def matrix_ddl(schema: str) -> str:
    columns = ", ".join(f"{name} {ddl}" for name, ddl, _ in MATRIX_COLUMNS)
    return f"CREATE TABLE {schema}.{MATRIX_TABLE} ({columns})"


def matrix_insert(schema: str) -> str:
    values = ", ".join(f"({literals})" for literals, _ in MATRIX_ROWS)
    return f"INSERT INTO {schema}.{MATRIX_TABLE} VALUES {values}"


@pytest.fixture(scope="module")
def schema(live_config: RedshiftLiveConfig) -> str:
    schema, _ = live_config["XORQ_REDSHIFT_PROBE_TABLE"].split(".")
    return schema


def connect(live_config: RedshiftLiveConfig, schema: str) -> RedshiftBackend:
    # By reference, so nothing this connection builds records the password.
    return xo.redshift.connect(
        host="${XORQ_REDSHIFT_HOST}",
        port=int(live_config["XORQ_REDSHIFT_PORT"]),
        user="${XORQ_REDSHIFT_USER}",
        password="${XORQ_REDSHIFT_PASSWORD}",
        database=live_config["XORQ_REDSHIFT_DATABASE"],
        schema=schema,
    )


@pytest.fixture(scope="module")
def con(live_config: RedshiftLiveConfig, schema: str) -> RedshiftBackend:
    con = connect(live_config, schema)
    if MATRIX_TABLE not in con.list_tables(database=schema):
        con.disconnect()
        pytest.skip(
            f"{schema}.{MATRIX_TABLE} does not exist; create it with "
            f"{matrix_ddl(schema)}; {matrix_insert(schema)}"
        )
    yield con
    con.disconnect()


@pytest.fixture(params=["adbc", "psycopg"])
def read_path(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Force the read path, and record which one actually served each read.

    A read that returns the right rows proves nothing about ADBC by itself,
    because the psycopg fallback returns the same rows.
    """
    if request.param == "adbc":
        pytest.importorskip("adbc_driver_postgresql")
    else:
        monkeypatch.setattr(
            RedshiftBackend,
            "_adbc_unavailable_reason",
            lambda self: "forced off by the test",
        )
    record = {"path": request.param, "adbc_opens": 0, "psycopg_reads": 0}
    real_open = RedshiftBackend._open_adbc_conn_or_none
    real_gen_name = ibis_util.gen_name

    def spy_open(self: RedshiftBackend) -> Any:
        conn = real_open(self)
        record["adbc_opens"] += conn is not None
        return conn

    def spy_gen_name(namespace: str) -> str:
        # Only the psycopg branch of to_pyarrow_batches names a cursor so.
        record["psycopg_reads"] += namespace == "postgres_cursor"
        return real_gen_name(namespace)

    monkeypatch.setattr(RedshiftBackend, "_open_adbc_conn_or_none", spy_open)
    monkeypatch.setattr(ibis_util, "gen_name", spy_gen_name)
    return record


def assert_served_by(record: dict[str, Any]) -> None:
    if record["path"] == "adbc":
        assert record["adbc_opens"] and not record["psycopg_reads"], record
    else:
        assert not record["adbc_opens"] and record["psycopg_reads"], record


def bind(con: RedshiftBackend, schema: str, entry: str) -> ir.Table:
    if entry == "table":
        return con.table(MATRIX_TABLE, database=schema)
    return con.sql(f"SELECT * FROM {schema}.{MATRIX_TABLE}")


def test_the_catalog_path_binds_every_column_as_declared(
    con: RedshiftBackend, schema: str
) -> None:
    schema_ = con.table(MATRIX_TABLE, database=schema).schema()

    assert dict(schema_.items()) == {name: dtype for name, _, dtype in MATRIX_COLUMNS}


def test_the_query_path_binds_every_column_as_the_catalog_path_does(
    con: RedshiftBackend, schema: str
) -> None:
    schema_ = bind(con, schema, "sql").schema()

    assert dict(schema_.items()) == {
        name: dtype.copy(nullable=True) for name, _, dtype in MATRIX_COLUMNS
    }


@pytest.mark.parametrize("entry", ENTRIES)
def test_every_value_reads_back_as_inserted(
    con: RedshiftBackend, schema: str, entry: str, read_path: dict[str, Any]
) -> None:
    t = bind(con, schema, entry)
    t = t.select(*(c for c in t.columns if c not in UNMAPPABLE)).order_by("id")

    rows = con.to_pyarrow(t).to_pylist()

    assert rows == EXPECTED_ROWS
    assert_served_by(read_path)


@pytest.mark.parametrize("entry", ENTRIES)
def test_an_unmappable_column_is_refused_on_read(
    con: RedshiftBackend, schema: str, entry: str
) -> None:
    t = bind(con, schema, entry).select("id", "sup")

    with pytest.raises(exc.UnmappableColumnError, match="'sup'"):
        con.to_pyarrow(t)


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_temporary_table_shadows_the_permanent_one_it_is_named_after(
    con: RedshiftBackend,
    live_config: RedshiftLiveConfig,
    schema: str,
    entry: str,
    read_path: dict[str, Any],
) -> None:
    """Unqualified, the name means the temporary table, on every path.

    Redshift resolves an unqualified name to a session's temporary table
    before the search path, and ADBC reads on a connection of its own, which
    cannot see that table.
    """
    shadowing = connect(live_config, schema)
    try:
        with shadowing.con.cursor() as cursor:
            cursor.execute(f"CREATE TEMPORARY TABLE {MATRIX_TABLE} (temp_only integer)")
            cursor.execute(f"INSERT INTO {MATRIX_TABLE} VALUES (42)")
        t = (
            shadowing.table(MATRIX_TABLE)
            if entry == "table"
            else shadowing.sql(f"SELECT * FROM {MATRIX_TABLE}")
        )

        assert t.columns == ("temp_only",)
        assert shadowing.to_pyarrow(t).to_pylist() == [{"temp_only": 42}]
        assert shadowing.table(MATRIX_TABLE, database=schema).columns == tuple(
            name for name, *_ in MATRIX_COLUMNS
        )
    finally:
        shadowing.disconnect()
