from pathlib import Path

from xorq.common.utils.caching_utils import get_xorq_cache_dir


def test_default_caching_dir():
    actual_dir = get_xorq_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, Path)

    assert actual_dir.match("**/.cache/xorq/")
