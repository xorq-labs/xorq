from __future__ import annotations

import pathlib

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.backends.mixpanel.tests.conftest import (
    env_ref_kwargs,
    fake_env,
)
from xorq.common.utils.env_utils import EnvConfigable
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.backends.profiles import (
    Profile,
    check_for_exposed_secrets,
    get_declared_secret_keys,
)


maybe_creds = EnvConfigable.subclass_from_kwargs(*fake_env).from_env()
have_live_creds = all(maybe_creds[name] for name in fake_env)


def test_declared_secret_keys() -> None:
    assert get_declared_secret_keys("mixpanel") == ("secret",)
    assert "password" in get_declared_secret_keys("postgres")
    assert get_declared_secret_keys("no-such-backend") is None


def test_check_for_exposed_secrets_uses_declared_keys() -> None:
    check_for_exposed_secrets("mixpanel", dict(env_ref_kwargs))
    with pytest.raises(ValueError, match="exposed secret keys: 'secret'"):
        check_for_exposed_secrets("mixpanel", dict(env_ref_kwargs, secret="raw"))
    # username is not a declared secret key
    check_for_exposed_secrets("mixpanel", dict(env_ref_kwargs, username="raw"))


def test_connect_requires_credentials() -> None:
    with pytest.raises(com.XorqError, match="requires"):
        xo.load_backend("mixpanel").connect(username="only-me")


def test_connect_preserves_env_refs(con: BaseBackend) -> None:
    assert con._profile.kwargs_dict["secret"] == env_ref_kwargs["secret"]
    assert con._client.secret == fake_env["MIXPANEL_SERVICE_ACCOUNT_SECRET"]


def test_profile_roundtrip(
    con: BaseBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    path = con._profile.save(alias="mixpanel-test")
    assert path.exists()
    loaded = Profile.load("mixpanel-test", profile_dir=tmp_path)
    assert loaded.hash_name == con._profile.hash_name
    assert loaded.get_con().list_tables() == ["engage", "events"]


def test_save_rejects_raw_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profile = Profile(
        con_name="mixpanel",
        kwargs_tuple=tuple(dict(env_ref_kwargs, secret="raw-secret").items()),
    )
    with pytest.raises(ValueError, match="exposed secret keys"):
        profile.save(alias="bad")


def test_resources(con: BaseBackend) -> None:
    assert con.list_tables() == ["engage", "events"]
    assert con.get_schema("events").names == (
        "event",
        "time",
        "distinct_id",
        "insert_id",
        "properties",
    )
    with pytest.raises(com.XorqError, match="no resource 'nope'"):
        con.get_schema("nope")
    with pytest.raises(com.XorqError, match="no resource 'nope'"):
        con.table("nope")


def test_read_events_is_deferred(con: BaseBackend) -> None:
    # no network at construction: fake credentials suffice
    expr = con.read_events("2026-07-01", "2026-07-07")
    assert expr.schema() == con.get_schema("events")
    assert con.table(
        "events", from_date="2026-07-01", to_date="2026-07-07"
    ).schema() == con.get_schema("events")
    assert con.read_engage().schema() == con.get_schema("engage")


def test_expr_construction_rejects_raw_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con = xo.load_backend("mixpanel").connect(
        username="user", secret="raw-secret", project_id=1
    )
    with pytest.raises(ValueError, match="exposed secret keys"):
        con.read_events("2026-07-01", "2026-07-07")


def test_read_only(con: BaseBackend) -> None:
    with pytest.raises(com.XorqError, match="read-only"):
        con.create_table("t", None)
    with pytest.raises(com.XorqError, match="read-only"):
        con.drop_table("t")


@pytest.mark.skipif(not have_live_creds, reason="live mixpanel creds not in env")
def test_live_read_events() -> None:
    con = xo.load_backend("mixpanel").connect(**env_ref_kwargs)
    df = con.read_events("2026-07-01", "2026-07-07").execute()
    assert not df.empty
    assert set(df.columns) == set(con.get_schema("events").names)
