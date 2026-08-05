from __future__ import annotations

import contextlib
import importlib
import pathlib
import sys
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pandas.testing as tm

from xorq.loader import _load_entry_points


reduction_tolerance = 1e-7


def write_fake_dist(root: pathlib.Path, con_name: str) -> None:
    """Write a minimal installed distribution declaring a xorq.backends entry
    point, so adding `root` to sys.path is indistinguishable from installing a
    backend plugin."""
    dist_info = root / f"xorq_{con_name}-0.1.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: xorq-{con_name}\nVersion: 0.1\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[xorq.backends]\n{con_name} = xorq_{con_name}\n"
    )


@contextlib.contextmanager
def installed_mid_process(root: pathlib.Path, con_name: str) -> Iterator[str]:
    """Install a backend distribution into this live process, with an entry-point
    cache that predates it -- i.e. `pip install` in a Jupyter kernel."""
    write_fake_dist(root, con_name)
    assert not any(ep.name == con_name for ep in _load_entry_points())  # warm it
    sys.path.insert(0, str(root))
    try:
        # the cached value still predates the install
        assert not any(ep.name == con_name for ep in _load_entry_points())
        yield con_name
    finally:
        sys.path.remove(str(root))
        importlib.invalidate_caches()
        _load_entry_points.cache_clear()


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


def reader_counts(expr: Any) -> list[int]:
    from xorq.expr.remote_table_exec import count_remote_table_readers  # noqa: PLC0415

    return sorted(count_remote_table_readers(expr).values())
