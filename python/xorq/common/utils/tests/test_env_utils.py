import os
from pathlib import Path

import pytest

from xorq.common.utils.env_utils import (
    EnvConfigable,
    parse_env_file,
)


def test_subclass_from_kwargs(monkeypatch):
    monkeypatch.setitem(os.environ, "X", "1")
    monkeypatch.delitem(os.environ, "SHOULDNT_EXIST", raising=False)

    EnvConfig = EnvConfigable.subclass_from_kwargs("X", "SHOULDNT_EXIST")
    env_config = EnvConfig.from_env()
    assert env_config.X == "1"
    assert env_config.SHOULDNT_EXIST == ""

    # env vars take priority
    env_config = EnvConfig.from_env(X=2)
    assert env_config.X == "1"
    assert env_config.SHOULDNT_EXIST == ""

    # X is an env var so will be passed as a string but must be an int
    EnvConfig = EnvConfigable.subclass_from_kwargs("SHOULDNT_EXIST", X=1)
    with pytest.raises(TypeError):
        env_config = EnvConfig.from_env()

    # so when using from_env, non-string fields must not exist in the env
    monkeypatch.delitem(os.environ, "X")
    EnvConfig = EnvConfigable.subclass_from_kwargs("SHOULDNT_EXIST", X=1)
    env_config = EnvConfig.from_env()
    assert env_config.X == 1
    assert env_config.SHOULDNT_EXIST == ""


def make_EnvConfig_from_content(tmp_path, content):
    env_file = Path(tmp_path).joinpath(".env")
    env_file.write_text(content)
    EnvConfig = EnvConfigable.subclass_from_env_file(env_file)
    return EnvConfig


def test_subclass_from_env_file(monkeypatch, tmp_path):
    EnvConfig = make_EnvConfig_from_content(
        tmp_path,
        "\n".join(
            (
                "export X=2",
                "Y=3",
                "a line that does not match, perhaps we should raise?",
            )
        ),
    )

    env_config = EnvConfig()
    assert env_config.X == "2"
    assert env_config.Y == "3"

    env_config = EnvConfig.from_env()
    assert env_config.X == "2"
    assert env_config.Y == "3"

    monkeypatch.setitem(os.environ, "X", "1")
    env_config = EnvConfig.from_env()
    assert env_config.X == "1"
    assert env_config.Y == "3"


def test_single_line_env(tmp_path):
    EnvConfig = make_EnvConfig_from_content(
        tmp_path,
        "x=1",
    )
    env_config = EnvConfig()
    assert env_config.x == "1"


def test_parse_multiline_env_vars(fixture_dir):
    multiline_env_vars_path = fixture_dir.joinpath("multiline_env_vars.env")
    dct = parse_env_file(multiline_env_vars_path)
    actual = {k: len(v.split("\n")) for k, v in dct.items()}
    expected = {
        k: int(v)
        for k, v in (el.split("=") for el in dct["names_to_numlines"].split(","))
    }
    assert actual == expected
