"""ADR-2213's invariant, pinned against the plugin contract itself.

These tests run the `fixture_backend` module through the full out-of-tree
path -- entry-point discovery mid-process, tier-3 secret keys with no mirror
entry, expression capture via `expr_safe_profile_kwargs` -- so the guarantee
a real REST plugin (e.g. xorq-mixpanel) depends on is enforced here, in the
tree that provides it, with no vendor integration shipped to do so.
"""

from __future__ import annotations

import base64
import pathlib
import re
import secrets
from collections.abc import Iterator

import pytest

import xorq.api as xo
from xorq.common.utils.attr_utils import secret_field_names
from xorq.ibis_yaml.compiler import build_expr
from xorq.tests.fixture_backend import FakeApiClient
from xorq.tests.util import installed_mid_process
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.backends.profiles import (
    con_name_to_secret_keys,
    get_secret_keys,
)


env_ref_kwargs = {
    "username": "${FAKEAPI_USERNAME}",
    "secret": "${FAKEAPI_SECRET}",
}
fixture_module = "xorq.tests.fixture_backend"


@pytest.fixture
def fake_env() -> dict[str, str]:
    # generated per test so the resolved values cannot appear in any repo
    # file: build_metadata.json embeds the working tree's uncommitted diff,
    # so a literal committed here would grep as a "leak" whenever this file
    # (or anything quoting it) has local modifications
    return {
        "FAKEAPI_USERNAME": f"fake-user-{secrets.token_hex(8)}",
        "FAKEAPI_SECRET": f"fake-secret-{secrets.token_hex(8)}",
    }


@pytest.fixture
def con(
    fake_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[BaseBackend]:
    for name, value in fake_env.items():
        monkeypatch.setenv(name, value)
    root = tmp_path_factory.mktemp("fakeapi-dist")
    with installed_mid_process(root, "fakeapi", module=fixture_module):
        yield xo.load_backend("fakeapi").connect(**env_ref_kwargs)


def test_secret_keys_come_from_the_hook_alone(con: BaseBackend) -> None:
    """The out-of-tree tier is sufficient: no mirror entry exists for the
    fixture backend -- an out-of-tree backend can never have one -- and the
    dynamic hook still widens the checked keys past the default."""
    assert "fakeapi" not in con_name_to_secret_keys
    assert get_secret_keys("fakeapi", {}) == ("password", "secret")


def test_expr_construction_rejects_raw_secret(
    tmp_path: pathlib.Path,
) -> None:
    with installed_mid_process(tmp_path, "fakeapi", module=fixture_module):
        con = xo.load_backend("fakeapi").connect(username="user", secret="raw-secret")
        with pytest.raises(ValueError, match="exposed secret keys: 'secret'"):
            con.read_records("q")


def test_expr_client_holds_references_not_resolved_values(
    con: BaseBackend, fake_env: dict[str, str]
) -> None:
    """`expr_safe_profile_kwargs` hands back the authored references while the
    interactive client holds the resolved (and repr-suppressed) values."""
    assert con.expr_safe_profile_kwargs() == env_ref_kwargs
    assert con._client.secret == fake_env["FAKEAPI_SECRET"]
    assert fake_env["FAKEAPI_SECRET"] not in repr(con._client)
    assert set(secret_field_names(FakeApiClient)) >= set(
        get_secret_keys("fakeapi", {})
    ) - {"password"}


def test_built_artifact_carries_no_resolved_credential(
    con: BaseBackend, fake_env: dict[str, str], tmp_path: pathlib.Path
) -> None:
    """The invariant, checked against the bytes on disk rather than asserted.

    The expression captures a client inside base64 cloudpickle bytes in
    `expr.yaml`, which no reviewer greps. A regression as small as handing the
    deferred read the resolved client instead of the reference-holding one
    leaks every credential into every artifact built from it, and would pass
    every other test in the suite. So the artifact is built and searched,
    base64 payloads decoded, for the resolved values -- and the references are
    asserted present, so a build that simply stopped capturing the client
    could not pass by carrying nothing at all.
    """
    expr = con.read_records("select 1")
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
