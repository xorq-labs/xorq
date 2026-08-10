from __future__ import annotations

import base64
import pathlib
import re

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.backends.mixpanel.client import MixpanelClient
from xorq.backends.mixpanel.tests.conftest import (
    env_ref_kwargs,
    fake_env,
)
from xorq.common.utils.attr_utils import secret_field_names
from xorq.common.utils.env_utils import EnvConfigable
from xorq.ibis_yaml.compiler import build_expr
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.backends.profiles import (
    Profile,
    check_for_exposed_secrets,
    con_name_to_secret_keys,
    get_dynamic_secret_keys,
)


maybe_creds = EnvConfigable.subclass_from_kwargs(*fake_env).from_env()
have_live_creds = all(maybe_creds[name] for name in fake_env)


def test_declared_secret_keys() -> None:
    # mixpanel declares its keys statically, so they answer from the mirror; the
    # dynamic hook arrives with the rest backend and returns None until then.
    assert con_name_to_secret_keys["mixpanel"] == ("secret",)
    assert "password" in con_name_to_secret_keys["postgres"]
    assert get_dynamic_secret_keys("mixpanel") is None
    assert get_dynamic_secret_keys("no-such-backend") is None


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
    # page_size wraps process_df in a partial; make_udxf reads its __name__
    assert con.read_engage(page_size=100).schema() == con.get_schema("engage")


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


def test_built_artifact_carries_no_resolved_credential(
    con: BaseBackend, tmp_path: pathlib.Path
) -> None:
    """The invariant, checked against the bytes on disk rather than asserted.

    This ADR's central claim is that a build artifact carries env-var
    *references* and never credential values. The dangerous half of that claim
    is unauditable by eye: the expression captures a client inside base64
    cloudpickle bytes in `expr.yaml`, which no reviewer greps and no other test
    reads. A regression as small as handing the deferred `Read` the resolved
    client instead of the reference-holding one leaks every credential into
    every artifact built from it, and would otherwise pass this whole suite.

    So the artifact is built and searched, base64 payloads decoded, for the
    resolved values -- and the references are asserted present, so that a build
    which simply stopped capturing the client could not pass by carrying
    nothing at all.
    """
    expr = con.read_events("2026-07-01", "2026-07-07")
    build_expr(expr, builds_dir=tmp_path)

    resolved = tuple(fake_env.values())
    referenced = tuple(env_ref_kwargs.values())
    found_reference = False

    for path in tmp_path.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        haystacks = [raw]
        # Decode every base64-looking run: a credential inside a cloudpickled
        # client is invisible to a plain grep of the artifact, which is exactly
        # the leak class this guards.
        for blob in re.findall(rb"[A-Za-z0-9+/=]{64,}", raw):
            try:
                haystacks.append(base64.b64decode(blob, validate=True))
            except Exception:
                continue
        for haystack in haystacks:
            for value in resolved:
                assert value.encode() not in haystack, (
                    f"{path.relative_to(tmp_path)} carries the resolved value for "
                    f"a credential; build artifacts must carry env-var references"
                )
            found_reference = found_reference or any(
                ref.encode() in haystack for ref in referenced
            )

    assert found_reference, (
        "no env-var reference found in the artifact, so this test proved "
        "nothing -- the expression is no longer capturing the client"
    )


def test_secrecy_mechanisms_agree(con: BaseBackend) -> None:
    """The tree has two disjoint secrecy mechanisms; this is the seam that makes
    them check each other.

    `secret_field` suppresses `repr` on the client, which holds RESOLVED
    credentials. The profile machinery enforces env-var *references* in a saved
    profile, via the secret keys `check_for_exposed_secrets` reads. Nothing
    connects them, so they can drift: a credential added to the profile's keys
    but declared with a plain `field()` would be profile-enforced and still
    printable in every traceback.

    The containment is one-directional. Everything the profile machinery calls a
    secret must ALSO be unprintable where its resolved value lives. The reverse
    does not hold and must not be asserted -- `username` is repr-suppressed as
    half a basic-auth credential while deliberately not profile-enforced, and
    that asymmetry is pinned below so it reads as a decision, not a gap.
    """
    unprintable = set(secret_field_names(MixpanelClient))
    enforced = set(con_name_to_secret_keys["mixpanel"])
    assert enforced <= unprintable, (
        f"profile-enforced but printable: {sorted(enforced - unprintable)}"
    )
    # the deliberate asymmetry, stated rather than left to inference
    assert "username" in unprintable and "username" not in enforced
    assert fake_env["MIXPANEL_SERVICE_ACCOUNT_SECRET"] not in repr(con._client)
