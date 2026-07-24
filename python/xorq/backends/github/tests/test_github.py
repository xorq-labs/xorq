from __future__ import annotations

import pathlib

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.common.utils.env_utils import EnvConfigable
from xorq.expr.relations import Read
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
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
