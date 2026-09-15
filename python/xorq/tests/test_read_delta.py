"""``read_delta`` on the embedded backend, against the deltalake floor.

This lives in the core suite, not under ``backends/datafusion/tests``, so that
ci-test-lowest-direct exercises the ``deltalake>=1.0`` floor -- and the
``write_deltalake`` / ``DeltaTable.to_pyarrow_dataset`` surface xorq calls --
without having to run the whole datafusion backend suite. See issue #2109.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from deltalake import write_deltalake

import xorq.api as xo


@pytest.fixture
def delta_path(tmp_path):
    path = tmp_path / "delta"
    write_deltalake(str(path), pd.DataFrame({"k": ["x"], "v": [1]}))
    return path


def test_read_delta(delta_path) -> None:
    con = xo.connect()

    assert con.read_delta(delta_path, "a").execute().to_dict("records") == [
        {"k": "x", "v": 1}
    ]


def test_read_delta_generates_table_name(delta_path) -> None:
    con = xo.connect()

    t = con.read_delta(delta_path)

    assert "read_delta" in t.op().name
    assert t.op().name in con.list_tables()


def test_read_delta_without_deltalake(tmp_path, monkeypatch) -> None:
    """The install hint fires when ``deltalake`` is not importable."""
    con = xo.connect()
    monkeypatch.setitem(sys.modules, "deltalake", None)

    with pytest.raises(ImportError, match="pip install deltalake"):
        con.read_delta(tmp_path / "delta", "a")
