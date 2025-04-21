import os
from pathlib import Path

import pytest

from xorq.common.utils.env_utils import EnvConfigable


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


def test_subclass_from_env_file(monkeypatch, tmp_path):
    content = "\n".join(
        (
            "export X=2",
            "Y=3",
            "a line that doesn't match, perhaps we should raise?",
        )
    )
    env_file = Path(tmp_path).joinpath(".env")
    env_file.write_text(content)
    EnvConfig = EnvConfigable.subclass_from_env_file(env_file)

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
