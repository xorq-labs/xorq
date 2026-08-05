from __future__ import annotations

import pathlib
import sys
import types

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
import xorq.vendor.ibis.backends.profiles as profiles_mod
from xorq.common.utils.env_utils import EnvConfigable
from xorq.expr.relations import Read
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)
from xorq.vendor.ibis.backends.profiles import (
    Profile,
    get_dynamic_secret_keys,
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
    still fully checked -- by both tiers, for different reasons.

    The dynamic tier answers because resolution imports the backend rather than
    inspecting `sys.modules` alone; an earlier draft skipped it here, and a
    credential the config named but the mirror did not saved as plaintext. The
    mirror answers without importing anything, and is what remains if the
    backend's extra is not installed at all.
    """
    monkeypatch.delitem(sys.modules, "xorq.backends.github", raising=False)
    assert "token" in get_dynamic_secret_keys("github")  # imported to answer
    assert "token" in get_secret_keys("github")  # and mirrored besides

    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profile = Profile(con_name="github", kwargs_tuple=(("token", "raw-token"),))
    with pytest.raises(ValueError, match="exposed secret keys: 'token'"):
        profile.save(alias="bad")


def test_mirror_answers_when_the_backend_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """With the extra absent the dynamic tier cannot answer at all. It degrades
    with a warning, and the static mirror is the only thing still checking."""
    monkeypatch.delitem(sys.modules, "xorq.backends.github", raising=False)
    real = profiles_mod._load_entry_points()

    def unimportable():
        def load():
            raise ImportError("No module named 'github_extra'")

        return tuple(
            types.SimpleNamespace(name=ep.name, module=ep.module, load=load)
            if ep.name == "github"
            else ep
            for ep in real
        )

    monkeypatch.setattr(profiles_mod, "_load_entry_points", unimportable)
    with pytest.warns(RuntimeWarning, match="could not import the backend"):
        assert get_dynamic_secret_keys("github") is None
    assert "token" in get_secret_keys("github")  # the mirror still answers

    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    profile = Profile(con_name="github", kwargs_tuple=(("token", "raw-token"),))
    with pytest.raises(ValueError, match="exposed secret keys: 'token'"):
        profile.save(alias="bad")


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
