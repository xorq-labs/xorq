from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import xorq.api as xo
from xorq.tests.util import assert_frame_equal


if TYPE_CHECKING:
    from pathlib import Path

    from xorq.backends.bigquery import Backend


# the google client libraries are an optional (`--extra bigquery`) dependency
pytest.importorskip("google.cloud.bigquery")


@pytest.mark.bigquery
def test_build_run_roundtrip(con: Backend, temp_table: str, tmp_path: Path) -> None:
    # build the expr to disk (ibis_yaml), load it back, and confirm the loaded
    # expr executes to the same result as the original
    df = pd.DataFrame({"playerID": ["a", "b"], "yearID": [2015, 2016], "G": [1, 2]})
    con.create_table(temp_table, obj=df)
    expr = (
        con.table(temp_table).filter(lambda t: t.yearID == 2015).select("playerID", "G")
    )

    build_path = xo.build_expr(expr, builds_dir=tmp_path / "builds")
    loaded = xo.load_expr(build_path)

    expected = expr.execute()
    result = loaded.execute()
    assert not result.empty
    assert_frame_equal(result, expected)
