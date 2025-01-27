from typing import Any

import pandas as pd
import pandas.testing as tm


reduction_tolerance = 1e-7


def assert_series_equal(
    left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
) -> None:
    kwargs.setdefault("check_dtype", True)
    kwargs.setdefault("check_names", False)
    tm.assert_series_equal(left, right, *args, **kwargs)


def assert_frame_equal(
    left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
) -> None:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    kwargs.setdefault("check_dtype", True)
    tm.assert_frame_equal(left, right, *args, **kwargs)


def default_series_rename(series: pd.Series, name: str = "tmp") -> pd.Series:
    return series.rename(name)
