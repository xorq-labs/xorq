"""``read_delta``, against the deltalake floor.

This lives in the core suite, not under ``backends/*/tests``, so that
ci-test-lowest-direct exercises the ``deltalake>=1.0`` floor -- and the
``write_deltalake`` / ``DeltaTable.to_pyarrow_dataset`` surface xorq calls --
without having to run a whole backend suite. See issue #2109.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import xorq.api as xo


write_deltalake = pytest.importorskip("deltalake").write_deltalake


@pytest.fixture(params=["xorq", "datafusion"])
def con(request):
    connect = {"xorq": xo.connect, "datafusion": xo.datafusion.connect}[request.param]
    return connect()


@pytest.fixture
def delta_path(tmp_path):
    path = tmp_path / "delta"
    write_deltalake(str(path), pd.DataFrame({"k": ["x"], "v": [1]}))
    return path


def test_read_delta(con, delta_path) -> None:
    assert con.read_delta(delta_path, "a").execute().to_dict("records") == [
        {"k": "x", "v": 1}
    ]


def test_read_delta_generates_table_name(con, delta_path) -> None:
    t = con.read_delta(delta_path)

    assert "read_delta" in t.op().name
    assert t.op().name in con.list_tables()


def test_read_delta_kwargs(con, delta_path) -> None:
    """``**kwargs`` reach ``DeltaTable``, so an older version is readable."""
    write_deltalake(
        str(delta_path), pd.DataFrame({"k": ["y"], "v": [2]}), mode="overwrite"
    )

    assert con.read_delta(delta_path, "a", version=0).execute().to_dict("records") == [
        {"k": "x", "v": 1}
    ]


def test_read_delta_without_deltalake(con, tmp_path, monkeypatch) -> None:
    """The install hint fires when ``deltalake`` is not importable."""
    monkeypatch.setitem(sys.modules, "deltalake", None)

    with pytest.raises(ImportError, match="pip install deltalake"):
        con.read_delta(tmp_path / "delta", "a")
