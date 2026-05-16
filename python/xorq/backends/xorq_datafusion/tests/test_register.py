import pandas as pd
import pytest

import xorq.api as xo


pytestmark = pytest.mark.xorq_datafusion


def test_no_name_register():
    xo.connect().register(pd.DataFrame({"a": [1]}))
