"""The GitHub REST API as a xorq backend: the contract's N=2 validation.

Deliberately boring — header-link pagination, bearer auth, zero override
code. If this backend needs anything beyond config, the RestBackend
contract is wrong (ADR-0018's contract-validation obligation).
"""

from __future__ import annotations

from xorq.backends.rest import RestBackend
from xorq.backends.rest.config import (
    AuthConfig,
    ParamSpec,
    ResourceConfig,
    RestBackendConfig,
)
from xorq.vendor import ibis


__all__ = [
    "Backend",
]


issues_schema = ibis.schema(
    {
        "number": "int64",
        "title": "string",
        "state": "string",
        "created_at": "string",
        "user": "string",
        "properties": "string",
    }
)
repo_schema = ibis.schema(
    {
        "full_name": "string",
        "default_branch": "string",
        "stargazers_count": "int64",
        "open_issues_count": "int64",
        "properties": "string",
    }
)

GITHUB_CONFIG = RestBackendConfig(
    base_urls={"default": "https://api.github.com"},
    # unauthenticated requests are allowed (rate-limited to 60/hr), so the
    # token is optional; token_field defaults to the single declared field
    auth=AuthConfig(kind="bearer", fields=("token",), optional_fields=("token",)),
    resources=(
        ResourceConfig(
            name="issues",
            schema=issues_schema,
            path="/repos/{owner}/{repo}/issues",
            residual_column="properties",
            paginator="header_link",
            params=(
                ParamSpec("owner", required=True),
                ParamSpec("repo", required=True),
                ParamSpec("state"),
                ParamSpec("per_page"),
            ),
        ),
        ResourceConfig(
            name="repo",
            schema=repo_schema,
            path="/repos/{owner}/{repo}",
            residual_column="properties",
            params=(
                ParamSpec("owner", required=True),
                ParamSpec("repo", required=True),
            ),
        ),
    ),
)


class Backend(RestBackend):
    name = "github"
    config = GITHUB_CONFIG
