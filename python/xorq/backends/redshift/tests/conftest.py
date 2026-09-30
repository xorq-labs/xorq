"""Fixtures for the live-Redshift probe harness; the rules are documented in
``redshift_live_harness``."""

from __future__ import annotations

import pytest


psycopg = pytest.importorskip("psycopg")

import xorq.vendor.ibis as ibis  # noqa: E402
from xorq.backends.redshift.tests.redshift_live_harness import (  # noqa: E402
    PROBE_TABLE_SCHEMA,
    REQUIRED,
    RedshiftLiveConfig,
    refuse_unless_compute_select,
)


@pytest.fixture(scope="session")
def live_config():
    config = RedshiftLiveConfig.from_env()
    missing = [name for name in REQUIRED if not config.get(name)]
    if missing:
        pytest.skip(f"live Redshift probes need {missing}")
    return config


@pytest.fixture(scope="session")
def redshift_live(live_config):
    con = psycopg.connect(
        host=live_config["XORQ_REDSHIFT_HOST"],
        port=int(live_config["XORQ_REDSHIFT_PORT"]),
        dbname=live_config["XORQ_REDSHIFT_DATABASE"],
        user=live_config["XORQ_REDSHIFT_USER"],
        password=live_config["XORQ_REDSHIFT_PASSWORD"],
        autocommit=True,
        # Redshift reports UNICODE, which psycopg's codec map lacks, and has no
        # DEALLOCATE ALL; the backend's do_connect sets both for the same reasons.
        client_encoding="utf8",
        prepare_threshold=None,
    )
    yield con
    con.close()


@pytest.fixture(scope="session")
def probe_table(live_config):
    """The probe table as an unbound ibis table, for compiling probes."""
    database, name = live_config["XORQ_REDSHIFT_PROBE_TABLE"].split(".")
    return ibis.table(PROBE_TABLE_SCHEMA, name=name, database=database)


@pytest.fixture(scope="session")
def run_select(redshift_live):
    """Run one compute-node SELECT and return its rows."""

    def run(sql: str):
        refuse_unless_compute_select(sql)
        return redshift_live.execute(sql).fetchall()

    return run
