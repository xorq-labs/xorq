import pandas as pd

import xorq.api as xo


def test_memtable_round_trips(pg):
    # registering a memtable builds its INSERT from the inherited template
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = pg.to_pyarrow(xo.memtable(df)).to_pandas()
    assert result.equals(df)


def test_execute_memtable(pg):
    # execute converts through the vendored pandas converter: this runs in
    # the postgres CI job, which installs no `ibis` extra
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert pg.execute(xo.memtable(df)).equals(df)
