import pandas as pd

import xorq.api as xo


def test_no_name_register():
    xo.connect().register(pd.DataFrame({"a": [1]}))
