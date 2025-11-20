import re

import xorq.api as xo
from xorq.api import _
from xorq.common.utils.defer_utils import deferred_read_csv


def test_read(csv_dir):
    csv_name = "astronauts"
    csv_path = csv_dir / f"{csv_name}.csv"

    # we can work with a pandas expr without having read it yet
    pd_con = xo.pandas.connect()

    expr = deferred_read_csv(con=pd_con, path=csv_path, table_name=csv_name).filter(
        _.number > 6
    )

    pattern = r"Read\[name=astronauts, method_name=read_csv, source=pandas-\d+\]"

    assert re.search(pattern, repr(expr))
