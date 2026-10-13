import pandas as pd

import xorq.api as xo


def test_memtable_round_trips(pg):
    # registering a memtable builds its INSERT from the inherited template;
    # to_pyarrow rather than execute, which still imports the non-vendored
    # ibis converter and so needs the ibis extra
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = pg.to_pyarrow(xo.memtable(df)).to_pandas()
    assert result.equals(df)
