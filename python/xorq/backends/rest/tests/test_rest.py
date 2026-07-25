from __future__ import annotations

import json
import pathlib

import pytest

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.backends.rest import RestBackend
from xorq.backends.rest.config import (
    AuthConfig,
    ParamSpec,
    ResourceConfig,
    RestBackendConfig,
    schema_to_nullable_dtypes,
)
from xorq.backends.rest.engines import (
    NativeEngine,
    record_to_row,
)
from xorq.backends.rest.paginators import (
    HeaderLinkPaginator,
    OffsetPaginator,
    PageNumberPaginator,
    SinglePagePaginator,
    make_paginator,
)
from xorq.common.utils.dasher import tokenize
from xorq.expr.relations import Read
from xorq.vendor import ibis
from xorq.vendor.ibis.backends.profiles import (
    Profile,
    check_for_exposed_secrets,
    get_dynamic_secret_keys,
)


things_schema = ibis.schema({"id": "int64", "name": "string", "properties": "string"})
config_dict = {
    "base_urls": {"default": "https://api.example.com"},
    "auth": {"kind": "bearer", "fields": ["token"]},
    "resources": [
        {
            "name": "things",
            "path": "/things/{bucket}",
            "record_path": "items",
            "paginator": "page_number",
            "schema": {"id": "int64", "name": "string", "properties": "string"},
            "params": [{"name": "bucket", "required": True}, {"name": "state"}],
        },
    ],
}


def make_config(things_path: str = "/things/{bucket}") -> RestBackendConfig:
    return RestBackendConfig(
        base_urls={"default": "https://api.example.com"},
        auth=AuthConfig(kind="none"),
        resources=(
            ResourceConfig(
                name="things",
                schema=things_schema,
                path=things_path,
                params=(ParamSpec("bucket", required=True),),
            ),
            ResourceConfig(name="other", schema=things_schema, path="/other"),
        ),
    )


def test_from_dict_round_trip() -> None:
    config = RestBackendConfig.from_dict(config_dict)
    (things,) = (r for r in config.resources if r.name == "things")
    assert things.schema == things_schema
    assert things.required_params == ("bucket",)
    assert things.paginator == "page_number"
    assert config.auth.effective_secret_fields == ("token",)


def test_from_dict_rejects_fetch_override() -> None:
    resource = dict(config_dict["resources"][0], fetch_override=lambda b: None)
    with pytest.raises(ValueError, match="code-path-only"):
        RestBackendConfig.from_dict(dict(config_dict, resources=[resource]))


def test_resource_content_hash_is_declarative() -> None:
    config = make_config()
    edited = make_config(things_path="/things/v2/{bucket}")
    (things, other) = config.resources
    (things2, other2) = edited.resources
    assert things.content_hash != things2.content_hash  # path edit changes identity
    assert other.content_hash == other2.content_hash  # sibling untouched
    # fetch_override is code, not identity
    overridden = ResourceConfig(
        name="other", schema=things_schema, path="/other", fetch_override=lambda b: None
    )
    assert overridden.content_hash == other.content_hash


def test_schema_to_nullable_dtypes() -> None:
    assert schema_to_nullable_dtypes(things_schema) == {
        "id": "Int64",
        "name": "string",
        "properties": "string",
    }


class FakeResponse:
    def __init__(
        self, records: list, links: dict | None = None, body: dict | None = None
    ) -> None:
        self._records = records
        self.links = links or {}
        self._body = body

    def json(self) -> object:
        return self._body if self._body is not None else self._records


def test_header_link_paginator() -> None:
    paginator = HeaderLinkPaginator()
    more = FakeResponse([], links={"next": {"url": "https://x/things?page=2"}})
    done = FakeResponse([], links={})
    assert paginator.next(more, [], "https://x/things", {}) == (
        "https://x/things?page=2",
        {},
    )
    assert paginator.next(done, [], "https://x/things", {}) is None


def test_offset_paginator() -> None:
    paginator = OffsetPaginator(limit=2)
    params = paginator.initial_params({})
    assert params == {"limit": 2, "offset": 0}
    full = [{"id": 1}, {"id": 2}]
    assert paginator.next(FakeResponse(full), full, "u", params) == (
        "u",
        {"limit": 2, "offset": 2},
    )
    short = [{"id": 3}]
    assert paginator.next(FakeResponse(short), short, "u", params) is None


def test_page_number_paginator() -> None:
    paginator = PageNumberPaginator()
    params = paginator.initial_params({})
    assert params == {"page": 1}
    assert paginator.next(FakeResponse([{}]), [{}], "u", params) == ("u", {"page": 2})
    assert paginator.next(FakeResponse([]), [], "u", params) is None


def test_make_paginator_unknown() -> None:
    with pytest.raises(ValueError, match="unknown paginator"):
        make_paginator("nope")


def test_record_to_row_nests_to_json() -> None:
    resource = ResourceConfig(name="things", schema=things_schema)
    row = record_to_row({"id": 1, "name": {"nested": True}, "extra": "x"}, resource)
    assert row["id"] == 1
    assert json.loads(row["name"]) == {"nested": True}
    # properties is the overflow column: only unmapped fields, no duplication
    props = json.loads(row["properties"])
    assert props == {"extra": "x"}
    assert "id" not in props and "name" not in props


def test_basic_auth_requires_role_fields() -> None:
    # mapping fields by position is a silent-swap hazard, so it is rejected
    with pytest.raises(ValueError, match="username_field and password_field"):
        AuthConfig(kind="basic", fields=("user", "pass"))
    with pytest.raises(ValueError, match="not one of auth.fields"):
        AuthConfig(
            kind="basic",
            fields=("user", "pass"),
            username_field="user",
            password_field="nope",
        )
    ok = AuthConfig(
        kind="basic",
        fields=("user", "pass"),
        username_field="user",
        password_field="pass",
    )
    assert ok.username_field == "user"


def test_bearer_token_field_defaults_to_single_field() -> None:
    assert AuthConfig(kind="bearer", fields=("token",)).resolved_token_field == "token"
    with pytest.raises(ValueError, match="requires token_field"):
        AuthConfig(kind="bearer", fields=("a", "b"))


class CuratedBackend(RestBackend):
    # reuses the registered "rest" entry-point name so Profile validation
    # passes; config-in-code like a curated subclass
    config = make_config()


class CuratedEditedBackend(RestBackend):
    config = make_config(things_path="/things/v2/{bucket}")


def test_curated_sibling_independence() -> None:
    con = CuratedBackend().connect()
    edited = CuratedEditedBackend().connect()
    # editing resource "things" changes its identity...
    assert tokenize(con.read("things", bucket="b")) != tokenize(
        edited.read("things", bucket="b")
    )
    # ...but not the sibling's (config-in-code: profile excludes the config)
    assert tokenize(con.read("other")) == tokenize(edited.read("other"))


class AcmePaginator(SinglePagePaginator):
    """A curated-backend paginator, registered by merging over the base."""


class ExtendedRegistryBackend(RestBackend):
    config = make_config()
    paginators = {**RestBackend.paginators, "acme.single": AcmePaginator}


def test_make_engine_threads_subclass_registries() -> None:
    con = ExtendedRegistryBackend().connect()
    engine = con._engine
    assert isinstance(engine, NativeEngine)
    # the subclass-declared paginator resolves through the engine's registry
    assert isinstance(
        make_paginator("acme.single", registry=engine.paginators), AcmePaginator
    )
    with pytest.raises(ValueError, match="unknown paginator"):
        make_paginator("acme.single")  # base registry is untouched


def test_read_is_a_read_op_and_validates_params() -> None:
    con = CuratedBackend().connect()
    expr = con.read("things", bucket="b")
    op = expr.op()
    assert isinstance(op, Read)
    assert op.method_name == "fetch_resource"
    assert dict(op.read_kwargs)["resource"] == "things"
    with pytest.raises(com.XorqError, match="requires params"):
        con.read("things")
    with pytest.raises(com.XorqError, match="does not accept params"):
        con.read("other", nope=1)
    with pytest.raises(com.XorqError, match="no resource or table"):
        con.read("nope")


def test_self_service_profile_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("EXAMPLE_TOKEN", "fake-token")
    monkeypatch.setattr(xo.options.profiles, "profile_dir", tmp_path)
    con = xo.load_backend("rest").connect(token="${EXAMPLE_TOKEN}", config=config_dict)
    assert con.list_tables() == ["things"]
    path = con._profile.save(alias="example")
    assert "base_urls" in path.resolve().read_text()
    loaded = Profile.load("example", profile_dir=tmp_path).get_con()
    assert loaded.list_tables() == ["things"]
    assert loaded.get_schema("things") == things_schema


def test_dynamic_secret_keys() -> None:
    assert get_dynamic_secret_keys("rest", {"config": config_dict}) == ("token",)
    check_for_exposed_secrets("rest", {"config": config_dict, "token": "${T}"})
    with pytest.raises(ValueError, match="exposed secret keys: 'token'"):
        check_for_exposed_secrets("rest", {"config": config_dict, "token": "raw"})


def test_self_service_raw_secret_rejected_at_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con = xo.load_backend("rest").connect(token="raw-token", config=config_dict)
    with pytest.raises(ValueError, match="exposed secret keys"):
        con.read("things", bucket="b")
