import sys
from pathlib import Path

from xorq.common.utils.caching_utils import user_cache_dir


def test_default_caching_dir():
    actual_dir = user_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, Path)

    expected_match = (
        "/AppData/Local/xorq/cache/" if sys.platform == "win32" else "/.cache/xorq/"
    )
    assert actual_dir.match(f"**{expected_match}")
