"""Read-only warehouse usage, replayed live as a least-privileged Redshift user.

Connects DIRECTLY as ``xorq_ro`` -- never an admin login switched with ``SET
SESSION AUTHORIZATION`` -- because the ADBC read path and ``run-cached`` each
open their own connection from the credentials, and an admin credential there
would pass silently where the field fails. The role mirrors a read-only warehouse user's:
USAGE on the schema and SELECT on its tables, and no CREATE or TEMP anywhere.

Credentials come only from the environment, by reference:
``XORQ_REDSHIFT_HOST`` and ``XORQ_REDSHIFT_RO_PASSWORD``. The connection is
opened with ``${...}`` references, so a build records the reference and never
the value. Unset, every test here skips.

Every read asserts WHICH PATH SERVED IT: at least one ADBC connection opened, as
``xorq_ro``, and zero psycopg fallbacks. A read that only returns rows proves
nothing, because the fallback returns the same rows.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# postgres_utils imports the ADBC PostgreSQL driver at module level, and
# every test here asserts that ADBC served the read.
pytest.importorskip("adbc_driver_postgresql")

import xorq.api as xo  # noqa: E402
import xorq.vendor.ibis.expr.datatypes as dt  # noqa: E402
import xorq.vendor.ibis.util as ibis_util  # noqa: E402
from xorq.backends.redshift import Backend as RedshiftBackend  # noqa: E402
from xorq.caching import ParquetCache, ParquetSnapshotCache  # noqa: E402
from xorq.caching.storage import ParquetStorage  # noqa: E402
from xorq.common.exceptions import (  # noqa: E402
    RedshiftFreshnessUnavailable,
    UnmappableColumnError,
    UnsupportedOperationError,
)
from xorq.common.utils.env_utils import maybe_substitute_env_var  # noqa: E402
from xorq.common.utils.postgres_utils import PgADBC  # noqa: E402
from xorq.ibis_yaml.compiler import build_expr  # noqa: E402
from xorq.vendor.ibis.expr import types as ir  # noqa: E402


pytestmark = pytest.mark.redshift

RO_USER = "xorq_ro"
HOST_VAR = "XORQ_REDSHIFT_HOST"
PASSWORD_VAR = "XORQ_REDSHIFT_RO_PASSWORD"
HOST_REF = "${" + HOST_VAR + "}"
PASSWORD_REF = "${" + PASSWORD_VAR + "}"
DATABASE = "xorq_test"
SCHEMA = "xorq_test"


@pytest.fixture(scope="module")
def ro_con() -> RedshiftBackend:
    try:
        maybe_substitute_env_var(HOST_REF)
        maybe_substitute_env_var(PASSWORD_REF)
    except KeyError as e:
        pytest.skip(f"{e.args[0]} is not set; run through the least-privilege driver")
    return xo.redshift.connect(
        host=HOST_REF,
        port=5439,
        user=RO_USER,
        password=PASSWORD_REF,
        database=DATABASE,
        schema=SCHEMA,
    )


@pytest.fixture
def offers(ro_con: RedshiftBackend) -> ir.Table:
    return ro_con.table("offers")


@pytest.fixture
def read_path(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record who each ADBC connection is, and every psycopg fallback."""
    record = {"adbc_users": [], "psycopg_fallbacks": 0}
    real_open = RedshiftBackend._open_adbc_conn_or_none
    real_gen_name = ibis_util.gen_name

    def spy_open(self: RedshiftBackend) -> Any:
        conn = real_open(self)
        if conn is None:
            record["adbc_users"].append(None)
        else:
            cur = conn.cursor()
            try:
                cur.execute("select current_user")
                record["adbc_users"].append(cur.fetchone()[0])
            finally:
                cur.close()
        return conn

    def spy_gen_name(namespace: str) -> str:
        # Only the psycopg fallback of to_pyarrow_batches names a cursor so.
        if namespace == "postgres_cursor":
            record["psycopg_fallbacks"] += 1
        return real_gen_name(namespace)

    monkeypatch.setattr(RedshiftBackend, "_open_adbc_conn_or_none", spy_open)
    monkeypatch.setattr(ibis_util, "gen_name", spy_gen_name)
    return record


def assert_adbc_served(record: dict[str, Any]) -> None:
    assert record["adbc_users"], "no read reached the ADBC branch"
    assert set(record["adbc_users"]) == {RO_USER}, record
    assert record["psycopg_fallbacks"] == 0, record


def cache_files(base: Path) -> dict[str, tuple[int, int]]:
    if not base.exists():
        return {}
    return {
        str(f.relative_to(base)): (f.stat().st_ino, f.stat().st_mtime_ns)
        for f in base.rglob("*")
        if f.is_file()
    }


def assert_second_execute_is_a_hit(
    cache: ParquetSnapshotCache,
    uncached: ir.Table,
    expr: ir.Table,
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    puts = []
    real_put = ParquetStorage.put

    def counting_put(self: ParquetStorage, key: str, *args: Any, **kwargs: Any) -> Any:
        puts.append(key)
        return real_put(self, key, *args, **kwargs)

    monkeypatch.setattr(ParquetStorage, "put", counting_put)
    assert not cache.exists(uncached)
    assert cache_files(base) == {}
    first = expr.execute()
    files_after_miss = cache_files(base)
    assert len(puts) == 1
    assert len(files_after_miss) == 1
    assert cache.exists(uncached)
    again = expr.execute()
    # put publishes by tmp-file rename, so a rewrite would change the inode.
    assert len(puts) == 1
    assert cache_files(base) == files_after_miss
    assert len(first) == len(again)
    return first


def test_session_is_the_restricted_role(ro_con: RedshiftBackend) -> None:
    with ro_con.raw_sql("select current_user, session_user") as cur:
        assert cur.fetchone() == (RO_USER, RO_USER)
    shape = f"""
    select
      (select usesuper from pg_user where usename = '{RO_USER}'),
      has_database_privilege('{RO_USER}', '{DATABASE}', 'CREATE'),
      has_database_privilege('{RO_USER}', '{DATABASE}', 'TEMP'),
      has_schema_privilege('{RO_USER}', '{SCHEMA}', 'USAGE'),
      has_schema_privilege('{RO_USER}', '{SCHEMA}', 'CREATE'),
      has_table_privilege('{RO_USER}', '{SCHEMA}.offers', 'SELECT'),
      has_table_privilege('{RO_USER}', '{SCHEMA}.offers', 'INSERT')
    """
    with ro_con.raw_sql(shape) as cur:
        assert cur.fetchone() == (False, False, False, True, False, True, False)


def test_adbc_accelerator_is_importable_and_credentialed(
    ro_con: RedshiftBackend,
) -> None:
    assert ro_con._adbc_unavailable_reason() is None


def test_table_binds(offers: ir.Table) -> None:
    assert {"id", "title", "is_live", "fee_rate"} <= set(offers.schema().names)


def test_list_tables(ro_con: RedshiftBackend) -> None:
    assert "offers" in ro_con.list_tables()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda con, t: con.sql(
                f"select id, title, is_live, fee_rate from {SCHEMA}.offers"
            ),
            id="con.sql",
        ),
        pytest.param(
            lambda con, t: t.aggregate(
                live_fees=t.fee_rate.sum(where=t.is_live),
                live_titles=t.title.nunique(where=t.is_live),
            ),
            id="sum-nunique-where",
        ),
        pytest.param(
            lambda con, t: t.mutate(
                rn=xo.row_number().over(
                    xo.window(group_by=t.is_live, order_by=t.fee_rate.desc())
                )
            ),
            id="row_number",
        ),
        pytest.param(
            lambda con, t: t.mutate(d=xo.date(2026, 1, 15)).select("id", "d").limit(3),
            id="date-literal",
        ),
    ],
)
def test_read_is_served_by_adbc_as_the_role(
    ro_con: RedshiftBackend,
    offers: ir.Table,
    read_path: dict[str, Any],
    build: Callable[[RedshiftBackend, ir.Table], ir.Table],
) -> None:
    assert len(build(ro_con, offers).execute()) > 0
    assert_adbc_served(read_path)


def test_snapshot_cache_with_local_storage(
    offers: ir.Table, read_path: dict[str, Any], tmp_path: Path
) -> None:
    cache = ParquetSnapshotCache.from_kwargs(source=xo.connect(), base_path=tmp_path)
    expr = offers.filter(offers.is_live).select("id", "title").cache(cache)
    assert len(expr.execute()) > 0
    assert_adbc_served(read_path)


def test_snapshot_cache_with_local_storage_hits(
    offers: ir.Table,
    read_path: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = ParquetSnapshotCache.from_kwargs(source=xo.connect(), base_path=tmp_path)
    uncached = offers.filter(offers.is_live).select("id", "fee_rate")
    assert_second_execute_is_a_hit(
        cache, uncached, uncached.cache(cache), tmp_path, monkeypatch
    )
    assert_adbc_served(read_path)


def test_into_local_then_snapshot_cache_hits(
    offers: ir.Table,
    read_path: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported pattern for a user who cannot write to the warehouse."""
    local_con = xo.connect()
    cache = ParquetSnapshotCache.from_kwargs(source=local_con, base_path=tmp_path)
    uncached = (
        offers.filter(offers.is_live).select("id", "fee_rate").into_backend(local_con)
    )
    assert_second_execute_is_a_hit(
        cache, uncached, uncached.cache(cache), tmp_path, monkeypatch
    )
    assert_adbc_served(read_path)


@pytest.mark.parametrize(
    "into_local",
    (pytest.param(False, id="direct"), pytest.param(True, id="into_backend")),
)
def test_parquet_cache_is_refused_before_any_read(
    offers: ir.Table, read_path: dict[str, Any], tmp_path: Path, into_local: bool
) -> None:
    """No Redshift change signal is safe to key on, so the freshness cache refuses.

    Including the ``into_backend`` pattern: the freshness key reaches
    the Redshift table through the RemoteTable either way.
    """
    local_con = xo.connect()
    cache = ParquetCache.from_kwargs(source=local_con, base_path=tmp_path)
    expr = offers.filter(offers.is_live).select("id", "fee_rate")
    if into_local:
        expr = expr.into_backend(local_con)
    with pytest.raises(RedshiftFreshnessUnavailable, match="ParquetSnapshotCache"):
        expr.cache(cache).execute()
    assert read_path["adbc_users"] == []
    assert cache_files(tmp_path) == {}


def test_distinct_on_is_refused_with_the_row_number_remedy(offers: ir.Table) -> None:
    """Redshift has no FIRST aggregate; the compiler refuses and names the fix."""
    with pytest.raises(UnsupportedOperationError, match="row_number"):
        offers.distinct(on="is_live").execute()


@pytest.mark.xfail(
    strict=True,
    reason="Redshift folds the auto-generated upper-case alias to lower case "
    "and the ADBC read path's per-batch cast rejects the mismatch",
)
def test_auto_aliased_count_is_served_by_adbc(
    offers: ir.Table, read_path: dict[str, Any]
) -> None:
    """``t.count()`` names its column ``CountStar(offers)``; Redshift returns
    ``countstar(offers)``. The fallback used to hide this on schema= connections."""
    assert offers.count().execute() > 0
    assert_adbc_served(read_path)


@pytest.mark.xfail(
    strict=True,
    reason="a reduce()'d 25-way union exceeds the local SQL parser's recursion limit",
)
def test_25_way_union_of_warehouse_rows(offers: ir.Table) -> None:
    """Triage section 5: a reduce()'d union of 25 selects, run locally."""
    local = offers.select("id", "fee_rate").into_backend(xo.connect())
    parts = [local.filter(local.id == i).mutate(k=xo.literal(i)) for i in range(25)]
    unioned = parts[0]
    for part in parts[1:]:
        unioned = unioned.union(part)
    assert len(unioned.execute()) > 0


UNMAPPABLE_TABLE = "unmappable_columns"


@pytest.fixture
def unmappable(ro_con: RedshiftBackend) -> ir.Table:
    """id INTEGER NOT NULL, payload SUPER NOT NULL, note SUPER,
    blob VARBYTE(16), name VARCHAR(32); rows (1, {"a": 1}, NULL, 0xab, 'a')
    and (2, [1, 2], "x", NULL, NULL)."""
    return ro_con.table(UNMAPPABLE_TABLE)


def test_a_table_with_super_columns_binds_as_the_role(unmappable: ir.Table) -> None:
    schema = unmappable.schema()
    assert schema["payload"] == dt.NamedUnknown(raw_type="super", nullable=False)
    assert schema["note"] == dt.NamedUnknown(raw_type="super", nullable=True)
    assert schema["blob"].is_binary()


def test_the_other_columns_read_through_adbc(
    unmappable: ir.Table, read_path: dict[str, Any]
) -> None:
    rows = (
        unmappable.select("id", "name", "blob")
        .order_by("id")
        .execute()
        .to_dict("records")
    )
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["name"] == "a"
    assert bytes(rows[0]["blob"]) == b"\xab"
    assert_adbc_served(read_path)


@pytest.mark.parametrize(
    ("build", "columns"),
    [
        pytest.param(lambda t: t.select("id", "payload"), ("payload",), id="payload"),
        pytest.param(lambda t: t, ("payload", "note"), id="whole-table"),
    ],
)
def test_returning_a_super_column_is_refused_naming_it(
    unmappable: ir.Table,
    read_path: dict[str, Any],
    build: Callable[[ir.Table], ir.Table],
    columns: tuple[str, ...],
) -> None:
    with pytest.raises(UnmappableColumnError, match="super") as info:
        build(unmappable).execute()
    assert set(info.value.columns) == set(columns)
    # Refused before any statement: no read connection was opened.
    assert read_path["adbc_users"] == []


def test_filtering_on_a_super_column_without_returning_it_works(
    unmappable: ir.Table, read_path: dict[str, Any]
) -> None:
    got = unmappable.filter(unmappable.note.notnull()).select("id").execute()
    assert got["id"].tolist() == [2]
    assert_adbc_served(read_path)


def test_into_local_of_the_other_columns_then_snapshot_cache_hits(
    unmappable: ir.Table,
    read_path: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``into_backend`` pattern, on a table that carries SUPER columns."""
    local_con = xo.connect()
    cache = ParquetSnapshotCache.from_kwargs(source=local_con, base_path=tmp_path)
    uncached = unmappable.select("id", "name").into_backend(local_con)
    assert_second_execute_is_a_hit(
        cache, uncached, uncached.cache(cache), tmp_path, monkeypatch
    )
    assert_adbc_served(read_path)


def test_con_sql_returning_a_super_column_is_refused(ro_con: RedshiftBackend) -> None:
    t = ro_con.sql(f"select id, payload from {SCHEMA}.{UNMAPPABLE_TABLE}")
    with pytest.raises(UnmappableColumnError):
        t.execute()


@pytest.mark.xfail(
    strict=True,
    reason="a bare .cache() stores in the warehouse and raises InsufficientPrivilege",
)
def test_default_cache_fails_before_querying_and_names_the_pattern(
    offers: ir.Table,
) -> None:
    with pytest.raises(Exception, match="into_backend"):
        offers.filter(offers.is_live).select("id").cache().execute()


def build_live_offers(offers: ir.Table, tmp_path: Path) -> Path:
    expr = offers.filter(offers.is_live).select("id", "fee_rate")
    return Path(build_expr(expr, builds_dir=tmp_path / "builds"))


def run_cached(build_path: Path, tmp_path: Path, *extra: str) -> Any:
    return subprocess.run(
        [
            "xorq",
            "run-cached",
            str(build_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-path",
            str(tmp_path / "out.parquet"),
            "--format",
            "parquet",
            *extra,
        ],
        capture_output=True,
    )


def test_run_cached_by_default_is_refused_and_names_the_flag(
    offers: ir.Table, tmp_path: Path
) -> None:
    """run-cached defaults to ParquetCache, which refuses a Redshift table."""
    result = run_cached(build_live_offers(offers, tmp_path), tmp_path)
    stderr = result.stderr.decode()
    assert result.returncode != 0
    assert "RedshiftFreshnessUnavailable" in stderr
    assert "--cache-type snapshot" in stderr


def test_run_cached_reconnects_as_the_role(offers: ir.Table, tmp_path: Path) -> None:
    expr = offers.filter(offers.is_live).select("id", "fee_rate")
    expected = len(expr.execute())
    build_path = build_live_offers(offers, tmp_path)
    profiles = (build_path / "profiles.yaml").read_text()
    # The only credential the subprocess can use is the role's, by reference.
    assert maybe_substitute_env_var(PASSWORD_REF) not in profiles
    assert PASSWORD_REF in profiles
    assert RO_USER in profiles
    result = run_cached(build_path, tmp_path, "--cache-type", "snapshot")
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    assert pq.read_table(tmp_path / "out.parquet").num_rows == expected


def test_arrow_table_round_trips_locally() -> None:
    """Guard against a broken local engine masquerading as a warehouse failure."""
    assert xo.connect().register(pa.table({"id": [1]}), "one").execute()[
        "id"
    ].tolist() == [1]


# --- a2fc6261 live pass: the caller's libpq settings on the ADBC connection ---
#
# PgADBC now forwards the caller's libpq keywords onto the ADBC URI. Offline
# tests parse them back out of the URI; these check that the ADBC driver's
# libpq and Redshift act on them.


def _ro_connect(**kwargs: Any) -> RedshiftBackend:
    return xo.redshift.connect(
        host=HOST_REF,
        port=5439,
        user=RO_USER,
        password=PASSWORD_REF,
        database=DATABASE,
        schema=SCHEMA,
        **kwargs,
    )


@pytest.fixture(scope="module")
def tls_con(ro_con: RedshiftBackend) -> RedshiftBackend:
    certifi = pytest.importorskip("certifi")
    return _ro_connect(sslmode="verify-full", sslrootcert=certifi.where())


def test_verify_full_read_is_served_by_adbc(
    tls_con: RedshiftBackend, read_path: dict[str, Any]
) -> None:
    settings = PgADBC(tls_con).settings
    assert settings["sslmode"] == "verify-full"
    assert len(tls_con.table("offers").select("id").execute()) > 0
    assert_adbc_served(read_path)


def test_verify_full_adbc_refuses_an_untrusted_root(
    tls_con: RedshiftBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control: the ADBC connection really verifies, rather than
    carrying sslmode in its URI and ignoring it. Only the ADBC side sees the
    wrong root, so psycopg's session is untouched."""
    bogus = tmp_path / "untrusted-root.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(tmp_path / "key.pem"),
            "-out",
            str(bogus),
            "-days",
            "1",
            "-subj",
            "/CN=untrusted-test-root",
        ],
        check=True,
        capture_output=True,
    )
    real = PgADBC.settings
    monkeypatch.setattr(
        PgADBC,
        "settings",
        property(lambda self: {**real.fget(self), "sslrootcert": str(bogus)}),
    )
    with pytest.raises(Exception, match="certificate verify failed"):
        tls_con._open_adbc_conn_or_none()
    with tls_con.raw_sql("select 1") as cur:
        assert cur.fetchone() == (1,)


@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_a_callers_timezone_option_reaches_the_adbc_session(
    ro_con: RedshiftBackend, tz: str
) -> None:
    """Redshift accepts TimeZone as a startup option, and a caller's
    ``options`` reaches the ADBC session beside the schema's search_path.

    Measured 2026-09-29: the psycopg session stays UTC whatever the caller
    asks for, because postgres's ``_post_connect`` runs ``SET TIMEZONE = UTC``
    after connecting -- so with a non-UTC TimeZone the two sessions disagree.
    This pins that divergence; it is a finding, not a contract."""
    con = _ro_connect(options=f"-c TimeZone={tz}")
    with con.raw_sql("select current_setting('timezone')") as cur:
        assert cur.fetchone() == ("UTC",)
    adbc = con._open_adbc_conn_or_none()
    cur = adbc.cursor()
    try:
        cur.execute(
            "select current_setting('timezone'), current_setting('search_path')"
            f" from {SCHEMA}.offers limit 1"
        )
        assert cur.fetchone() == (tz, SCHEMA)
    finally:
        cur.close()
        adbc.close()
