import zipfile
from pathlib import Path

import pytest

from xorq.catalog.exceptions import WheelCollisionError
from xorq.catalog.zip_utils import harvest_entry_from_zip, write_zip


WHEEL = "entry/pkg-0.1-py3-none-any.whl"


def test_a_differing_wheel_of_the_same_name_is_a_collision(tmp_path: Path) -> None:
    first = write_zip(tmp_path / "first.zip", {WHEEL: b"one"})
    second = write_zip(tmp_path / "second.zip", {WHEEL: b"two"})
    seen_wheels: dict = {}
    with zipfile.ZipFile(first) as zf:
        harvest_entry_from_zip(zf, tmp_path, "first", seen_wheels)
    with (
        zipfile.ZipFile(second) as zf,
        pytest.raises(WheelCollisionError, match="second"),
    ):
        harvest_entry_from_zip(zf, tmp_path, "second", seen_wheels)
