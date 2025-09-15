import pathlib

from xorq.ibis_yaml.config import (
    BuildConfig,  # Replace 'your_module' with the appropriate module name
)


def test_default_hash_length():
    config = BuildConfig()
    assert config.hash_length == 12, "Default hash_length should be 12"


def test_default_build_path():
    config = BuildConfig()
    expected_path = pathlib.Path.cwd() / "builds"
    assert config.build_path == expected_path


def test_set_build_path():
    config = BuildConfig()
    new_path = pathlib.Path.cwd() / "custom_builds"
    config.build_path = new_path
    assert config.build_path == new_path
