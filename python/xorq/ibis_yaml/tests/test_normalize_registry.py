"""Tests for the by-name normalize_method registry (fixes #2155).

Every claim here fails if the by-name mechanism regresses to cloudpickle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pytest

import xorq.api as xo
from xorq.common.exceptions import NormalizeMethodError
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.file_utils import (
    normalize_read_path_md5sum,
    normalize_read_path_stat,
)
from xorq.ibis_yaml import normalize_registry as nr
from xorq.ibis_yaml.common import serialize_callable
from xorq.ibis_yaml.compiler import ExprDumper, build_expr, load_expr


def _custom_normalize(path: str | Path) -> tuple:
    return (("custom", str(path)),)


def test_builtins_serialize_by_name() -> None:
    assert nr.serialize_normalize_method(normalize_read_path_stat) == {
        "kind": "named",
        "name": "read_path_stat",
    }
    assert nr.serialize_normalize_method(normalize_read_path_md5sum) == {
        "kind": "named",
        "name": "read_path_md5sum",
    }
    assert nr.serialize_normalize_method(None) == {"kind": "none"}


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(None, id="none"),
        pytest.param(normalize_read_path_stat, id="stat"),
        pytest.param(normalize_read_path_md5sum, id="md5sum"),
    ],
)
def test_roundtrip_identity(fn: Optional[Callable]) -> None:
    payload = nr.serialize_normalize_method(fn)
    # the same function object comes back -- no pickle, no re-import
    assert nr.deserialize_normalize_method(payload) is fn


def test_serialize_rejects_custom_callable() -> None:
    with pytest.raises(NormalizeMethodError, match="not a registered"):
        nr.serialize_normalize_method(_custom_normalize)


def test_validate_allows_none_and_builtins() -> None:
    for fn in (None, normalize_read_path_stat, normalize_read_path_md5sum):
        nr.validate(fn)  # no raise


def test_deserialize_unknown_named_key() -> None:
    with pytest.raises(NormalizeMethodError, match="newer or incompatible"):
        nr.deserialize_normalize_method({"kind": "named", "name": "bogus"})


def test_deserialize_unknown_kind() -> None:
    with pytest.raises(NormalizeMethodError, match="unknown normalize_method encoding"):
        nr.deserialize_normalize_method({"kind": "sideways"})


def test_legacy_bare_string_resolvable_still_loads() -> None:
    # pre-fix builds stored a bare base64 cloudpickle string
    encoded = serialize_callable(normalize_read_path_stat)
    assert nr.deserialize_normalize_method(encoded) is normalize_read_path_stat


def test_legacy_reserved_pickle_tag_resolvable_loads() -> None:
    encoded = serialize_callable(normalize_read_path_md5sum)
    payload = {"kind": "pickle", "pickle": encoded}
    assert nr.deserialize_normalize_method(payload) is normalize_read_path_md5sum


def test_legacy_missing_module_raises_catchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the 0.3.28 case: a legacy pickle whose module is absent must surface a
    # clean, catchable NormalizeMethodError -- not a bare ModuleNotFoundError.
    def boom(_encoded: str) -> Callable:
        raise ModuleNotFoundError("no module named ghost", name="ghost")

    monkeypatch.setattr("xorq.ibis_yaml.common.deserialize_callable", boom)
    with pytest.raises(NormalizeMethodError, match="ghost"):
        nr.deserialize_normalize_method("anything")


# --- lockdown at the two injection points -----------------------------------


def test_deferred_read_rejects_custom_normalize_method(tmp_path: Path) -> None:
    pq = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq)
    con = xo.connect()
    with pytest.raises(NormalizeMethodError, match="not a registered"):
        deferred_read_parquet(
            pq, con, table_name="t", normalize_method=_custom_normalize
        )


def test_deferred_read_builtins_accepted(tmp_path: Path) -> None:
    pq = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq)
    con = xo.connect()
    t = deferred_read_parquet(pq, con, table_name="t")
    assert t.op().normalize_method is normalize_read_path_stat
    # relocatable forces md5sum, still registry-resolvable
    t2 = deferred_read_parquet(pq, con, table_name="t2", relocatable=True)
    assert t2.op().normalize_method is normalize_read_path_md5sum


def test_expr_dumper_rejects_custom_read_normalize_method(tmp_path: Path) -> None:
    pq = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq)
    con = xo.connect()
    expr = deferred_read_parquet(pq, con, table_name="t")
    with pytest.raises(NormalizeMethodError, match="not a registered"):
        ExprDumper(expr, read_normalize_method=_custom_normalize)


# --- end-to-end build round-trip --------------------------------------------


def test_build_yaml_has_named_not_pickle(tmp_path: Path) -> None:
    pq = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq)
    con = xo.connect()
    expr = deferred_read_parquet(pq, con, table_name="t").filter(xo._.a > 1)

    build_path = build_expr(expr, builds_dir=str(tmp_path / "builds"))
    yaml_text = (build_path / "expr.yaml").read_text()

    assert "kind: named" in yaml_text
    # the by-name key is present; no base64 pickle blob for normalize_method
    assert "read_path" in yaml_text

    loaded = load_expr(build_path)
    assert loaded.execute().shape == (2, 1)
