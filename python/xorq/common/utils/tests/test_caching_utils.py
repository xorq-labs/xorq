from pathlib import Path

import pytest

import xorq.vendor.ibis as ibis
from xorq.common.utils.caching_utils import CacheKey, get_xorq_cache_dir
from xorq.vendor.ibis.expr.types.core import ExprKind, ExprMetadata


def test_default_caching_dir():
    actual_dir = get_xorq_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, Path)

    assert actual_dir.match("**/.cache/xorq/")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key": 123, "relative_path": "my_cache"},
        {"key": "abc", "relative_path": None},
    ],
)
def test_cache_key_rejects_non_str_fields(kwargs):
    with pytest.raises(TypeError):
        CacheKey(**kwargs)


def test_expr_metadata_cache_keys_rejects_non_cache_key_items():
    with pytest.raises(TypeError):
        ExprMetadata(
            kind=ExprKind.Source,
            schema_out=ibis.Schema({"x": "int64"}),
            cache_keys=("not_a_cache_key",),
        )
