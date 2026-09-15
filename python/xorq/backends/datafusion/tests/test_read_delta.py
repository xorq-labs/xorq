"""``read_delta`` on the datafusion backend."""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from deltalake import write_deltalake

import xorq.api as xo


def test_read_delta(tmp_path) -> None:
    con = xo.datafusion.connect()
    path = tmp_path / "delta"
    write_deltalake(str(path), pd.DataFrame({"k": ["x"], "v": [1]}))

    assert con.read_delta(path, "a").execute().to_dict("records") == [
        {"k": "x", "v": 1}
    ]


def test_read_delta_generates_table_name(tmp_path) -> None:
    con = xo.datafusion.connect()
    path = tmp_path / "delta"
    write_deltalake(str(path), pd.DataFrame({"k": ["x"], "v": [1]}))

    t = con.read_delta(path)

    assert "read_delta" in t.op().name
    assert t.op().name in con.list_tables()


def test_read_delta_without_deltalake(tmp_path, monkeypatch) -> None:
    """The install hint fires when ``deltalake`` is not importable."""
    con = xo.datafusion.connect()
    monkeypatch.setitem(sys.modules, "deltalake", None)

    with pytest.raises(ImportError, match="pip install deltalake"):
        con.read_delta(tmp_path / "delta", "a")
