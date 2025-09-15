from typing import Any

import pandas as pd
import pandas.testing as tm


reduction_tolerance = 1e-7


def _pandas_semi_join(left, right, on, **_):
    assert len(on) == 1, str(on)
    inner = pd.merge(left, right, how="inner", on=on)
    filt = left.loc[:, on[0]].isin(inner.loc[:, on[0]])
    return left.loc[filt, :]


def _pandas_anti_join(left, right, on, **_):
    inner = pd.merge(left, right, how="left", indicator=True, on=on)
    return inner[inner["_merge"] == "left_only"]


IMPLS = {
    "semi": _pandas_semi_join,
    "anti": _pandas_anti_join,
}


def check_eq(left, right, how, **kwargs):
    impl = IMPLS.get(how, pd.merge)
    return impl(left, right, how=how, **kwargs)


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
