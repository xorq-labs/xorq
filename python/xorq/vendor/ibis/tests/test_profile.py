import pathlib

import pytest

import xorq as xo
from xorq.vendor.ibis.backends import BaseBackend, Profile, Profiles


local_con_names = ("duckdb", "let", "datafusion", "pandas")
remote_connectors = (
    xo.postgres.connect_env,
    xo.postgres.connect_examples,
)
local_connectors = tuple(getattr(xo, con_name).connect for con_name in local_con_names)


@pytest.mark.parametrize(
    "con_name",
    local_con_names
    + (
        pytest.param(
            "invalid-con-name", marks=pytest.mark.xfail(reason="only valid con names")
        ),
    ),
)
def test_con_has_profile(con_name):
    con = getattr(xo, con_name).connect()
    assert isinstance(con, BaseBackend)
    profile = getattr(con, "_profile", None)
    assert isinstance(profile, Profile)
    assert profile.almost_equals(Profile.from_con(con))
    #
    other = profile.get_con()
    assert con.name == other.name
    # this doesn't work because _con_args, _con_kwargs doesn't get the defaults which are eventually invoked
    # assert hash(con) == hash(other)
    assert profile.almost_equals(other._profile)


@pytest.mark.parametrize("connect", remote_connectors)
def test_remote_con_works(connect):
    con = connect()
    assert isinstance(con, BaseBackend)
    profile = getattr(con, "_profile", None)
    assert isinstance(profile, Profile)
    assert profile.almost_equals(Profile.from_con(con))
    #
    other = profile.get_con()
    assert con.name == other.name
    # this doesn't work because _con_args, _con_kwargs doesn't get the defaults which are eventually invoked
    # assert hash(con) == hash(other)
    assert profile.almost_equals(other._profile)
    assert con.list_tables() == other.list_tables()


def test_profiles(monkeypatch, tmp_path):
    default_profile_dir = xo.options.profiles.profile_dir
    assert default_profile_dir == pathlib.Path("~/.config/letsql/profiles").expanduser()
    profiles = Profiles()
    assert profiles.profile_dir == default_profile_dir
    assert not profiles.list()

    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profiles = Profiles()
    assert profiles.profile_dir == tmp_path


@pytest.mark.parametrize("connector", remote_connectors + local_connectors)
def test_save_load(connector, monkeypatch, tmp_path):
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)

    con = connector()
    profiles = Profiles()
    profile = con._profile

    profile.save()

    others = tuple(
        (
            profiles.get(profile.hash_name),
            profiles[profile.hash_name],
            profile.load(profile.hash_name),
        )
    )
    for other in others:
        assert profile == other
        assert con.list_tables() == other.get_con().list_tables()
