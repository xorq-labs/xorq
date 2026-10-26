from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo


@pytest.mark.parametrize("con_name", ["sqlite_con", "persistent_sqlite_con"])
@pytest.mark.parametrize(
    ("data", "expected", "expected_type"),
    [
        pytest.param(
            pd.Series([1, None], dtype="Int64"), [1, None], pa.int64(), id="Int64-na"
        ),
        pytest.param(pd.Series([1, 2], dtype="Int64"), [1, 2], pa.int64(), id="Int64"),
        pytest.param(
            pd.Series([1, None], dtype="Int32"), [1, None], pa.int32(), id="Int32-na"
        ),
        pytest.param(
            pd.Series([1.5, None], dtype="Float64"),
            [1.5, None],
            pa.float64(),
            id="Float64-na",
        ),
        pytest.param(
            pd.Series([True, None], dtype="boolean"),
            [True, None],
            pa.bool_(),
            id="boolean-na",
        ),
        pytest.param(
            pd.Series(["a", None], dtype="string"),
            ["a", None],
            pa.string(),
            id="string-na",
        ),
        pytest.param(pd.Series([1, 2]), [1, 2], pa.int64(), id="int64"),
        pytest.param(
            pd.Series([1.5, np.nan]), [1.5, None], pa.float64(), id="float64-nan"
        ),
        pytest.param(
            pd.Series([datetime(2020, 1, 2, 3, 4, 5), None], dtype="datetime64[ns]"),
            [datetime(2020, 1, 2, 3, 4, 5), None],
            pa.timestamp("us"),
            id="datetime64-nat",
        ),
        pytest.param(
            pa.array([1, None], type=pa.int64()), [1, None], pa.int64(), id="arrow"
        ),
    ],
)
def test_memtable_nulls_and_values_roundtrip(
    request, con_name, data, expected, expected_type
):
    con = request.getfixturevalue(con_name)
    frame = (
        pa.table({"k": data})
        if isinstance(data, pa.Array)
        else pd.DataFrame({"k": data})
    )
    column = con.to_pyarrow(xo.memtable(frame))["k"]
    assert column.to_pylist() == expected
    assert column.type == expected_type
