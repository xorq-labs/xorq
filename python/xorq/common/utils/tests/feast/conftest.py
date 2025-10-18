import sys
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


pytest.importorskip("feast")


def freshen_driver_stats(
    store, end_date=None, delta=None, driver_entities=(1001, 1002, 1003, 1004, 1005)
):
    from feast.driver_test_data import create_driver_hourly_stats_df

    end_date = end_date or datetime.now().replace(microsecond=0, second=0, minute=0)
    delta = delta or timedelta(days=15)
    driver_df = create_driver_hourly_stats_df(
        list(driver_entities),
        end_date - delta,
        end_date,
    )
    parquet_path = store.path.joinpath("data", "driver_stats.parquet")
    parquet_path.parent.mkdir(exist_ok=True, parents=True)
    driver_df.to_parquet(parquet_path)


def make_store(fixture_dir, tmpdir):
    from xorq.common.utils.feast_replication_utils import (
        Store,
    )

    tgz_path = fixture_dir.joinpath("feature_repo.tgz")
    with tarfile.TarFile.gzopen(tgz_path) as tfh:
        tfh.extractall(tmpdir)
    store = Store(tmpdir)
    freshen_driver_stats(store)
    return store


def make_store_applied(fixture_dir, tmpdir):
    store_applied = make_store(fixture_dir, tmpdir)
    # we must directly change sys.path
    # # monkeypatch.syspath_prepend is only function scope
    old_sys_path, sys.path = sys.path, [str(tmpdir)] + sys.path
    store_applied.apply()
    sys.path = old_sys_path
    return store_applied


@pytest.fixture(scope="function")
def fresh_store(fixture_dir, tmpdir):
    fresh_store = make_store(fixture_dir, tmpdir)
    return fresh_store


@pytest.fixture(scope="session")
def store_applied(fixture_dir, tmpdir_factory):
    tmpdir = Path(tmpdir_factory.mktemp("tmp"))
    store_applied = make_store_applied(fixture_dir, tmpdir)
    return store_applied


@pytest.fixture(scope="session")
def store_applied_materialized(fixture_dir, tmpdir_factory):
    tmpdir = Path(tmpdir_factory.mktemp("tmp"))
    store_applied_materialized = make_store_applied(fixture_dir, tmpdir)
    store_applied_materialized.store.materialize_incremental(end_date=datetime.now())
    return store_applied_materialized
