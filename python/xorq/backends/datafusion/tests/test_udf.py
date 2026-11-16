from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.ibis_yaml.compiler import build_expr, load_expr


@pytest.mark.parametrize(
    ("config", "expected"),
    [({}, 3), ({"volatility": "immutable"}, 1)],
)
def test_pyarrow_udf_load_keeps_volatility(
    tmp_path: Path, config: dict, expected: int
) -> None:
    # on datafusion a missing volatility means volatile, so a loaded immutable
    # UDF that lost its config would stop folding
    @xo.udf.scalar.pyarrow(**config)
    def noise(arr: dt.float64) -> dt.float64:
        return pa.array([random.random() for _ in range(len(arr))])

    t = xo.datafusion.connect().create_table("t", {"x": [1.0, 2.0, 3.0]})
    expr = t.mutate(r=noise(xo.literal(1.0, "float64")))
    loaded = load_expr(build_expr(expr, builds_dir=tmp_path))

    assert loaded.execute().r.nunique() == expr.execute().r.nunique() == expected
