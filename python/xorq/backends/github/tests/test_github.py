from __future__ import annotations

import pathlib
import sys

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.common.utils.env_utils import EnvConfigable
from xorq.expr.relations import Read
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)
from xorq.vendor.ibis.backends.profiles import (
    Profile,
    get_declared_secret_keys,
    get_secret_keys,
)


maybe_token = EnvConfigable.subclass_from_kwargs("GITHUB_TOKEN").from_env()
have_token = bool(maybe_token["GITHUB_TOKEN"])


def test_connects_without_credentials() -> None:
    con = xo.load_backend("github").connect()
    assert con.list_tables() == ["issues", "repo"]
    assert con.get_schema("issues").names == (
        "number",
        "title",
        "state",
        "created_at",
        "user",
        "properties",
    )


def test_read_is_deferred_and_validates() -> None:
    con = xo.load_backend("github").connect()
    expr = con.read("issues", owner="xorq-labs", repo="xorq")
    assert isinstance(expr.op(), Read)
    assert dict(expr.op().read_kwargs)["resource"] == "issues"
    with pytest.raises(com.XorqError, match="requires params"):
        con.read("issues")


def test_build_artifact_is_declarative(tmp_path: pathlib.Path) -> None:
    con = xo.load_backend("github").connect()
    expr = con.read("repo", owner="xorq-labs", repo="xorq")
    build_path = pathlib.Path(build_expr(expr, builds_dir=tmp_path))
    expr_yaml = (build_path / "expr.yaml").read_text()
    assert "fetch_resource" in expr_yaml
    assert "pickle" not in expr_yaml.lower()
    assert "github" in (build_path / "profiles.yaml").read_text()
    loaded = load_expr(build_path)
    assert loaded.schema() == con.get_schema("repo")


def test_secret_keys_survive_an_unimported_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A profile hand-authored in a process that never imported the backend is
    still checked -- by the static mirror, since a profile with no config
    kwarg gives the declared sources nothing to resolve against.

    The sources themselves are mirrored too (`con_name_to_secret_key_sources`),
    so a config-carrying profile is covered without the import either; this
    test pins the no-config case, where the static mirror entry is the only
    tier that can answer -- without it this case would check `("password",)`
    alone, which matches none of github's fields, and a raw token would save
    with no error.
    """
    monkeypatch.delitem(sys.modules, "xorq.backends.github", raising=False)
    assert get_declared_secret_keys("github") == ()  # no config: nothing to read
    assert "token" in get_secret_keys("github")  # the mirror answers

    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profile = Profile(con_name="github", kwargs_tuple=(("token", "raw-token"),))
    with pytest.raises(ValueError, match="exposed secret keys: 'token'"):
        profile.save(alias="bad")


def test_a_config_override_names_its_own_secret_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Curated does not mean static-keys-only: `do_connect` accepts a
    `config=` override on any subclass, and a profile carrying one names its
    own credential kwargs. github's declared sources (inherited from
    RestBackend, mirrored per con_name) read them; a static-only split would
    check `('token',)` and let the renamed credential save."""
    config = {
        "base_urls": {"default": "https://api.github.com"},
        "auth": {"kind": "bearer", "fields": ["gh_pat"]},
    }
    assert get_declared_secret_keys("github", {"config": config}) == ("gh_pat",)
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profile = Profile(
        con_name="github",
        kwargs_tuple=(("config", config), ("gh_pat", "raw-pat-value")),
    )
    with pytest.raises(ValueError, match="exposed secret keys: 'gh_pat'"):
        profile.save(alias="bad")
    assert not tuple(tmp_path.iterdir())


@pytest.mark.skipif(not have_token, reason="GITHUB_TOKEN not in env")
def test_live_repo_and_paginated_issues() -> None:
    con = xo.load_backend("github").connect(token="${GITHUB_TOKEN}")
    repo = con.read("repo", owner="xorq-labs", repo="xorq").execute()
    assert repo.full_name.iloc[0] == "xorq-labs/xorq"
    issues = con.read(
        "issues", owner="xorq-labs", repo="xorq", state="open", per_page=50
    ).execute()
    # multi-page via header_link; cross-checked against the repo record
    assert len(issues) == repo.open_issues_count.iloc[0]
