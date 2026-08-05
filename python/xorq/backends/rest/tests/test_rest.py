from __future__ import annotations

import email.utils
import json
import pathlib
import warnings
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import attr
import pandas as pd
import pytest
import requests

import xorq.api as xo
import xorq.common.exceptions as com
from xorq.backends.rest import RestBackend
from xorq.backends.rest.config import (
    AuthConfig,
    ParamSpec,
    ResourceConfig,
    RestBackendConfig,
    identity_field_names,
    schema_to_nullable_dtypes,
)
from xorq.backends.rest.engines import (
    MAX_RETRY_WAIT,
    NativeEngine,
    record_to_row,
    retry_after_seconds,
    warn_on_absent_columns,
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


def make_config(
    things_path: str = "/things/{bucket}",
    base_url: str = "https://api.example.com",
    auth: AuthConfig | None = None,
) -> RestBackendConfig:
    return RestBackendConfig(
        base_urls={"default": base_url},
        auth=auth if auth is not None else AuthConfig(kind="none"),
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


def test_identity_field_membership_is_derived_from_the_declaration() -> None:
    """Identity membership must be derived from `attrs.fields`, so a field
    added later is identity-bearing by default. The only way out is an
    explicit metadata annotation -- these assertions are the tripwire that a
    new field cannot silently escape the hash."""
    excluded = {
        ResourceConfig: {"fetch_override"},  # code, not declarative config
        RestBackendConfig: {"resources"},  # folded per-resource instead
        AuthConfig: set(),
        ParamSpec: set(),
    }
    for cls, opted_out in excluded.items():
        declared = tuple(f.name for f in attr.fields(cls))
        assert identity_field_names(cls) == tuple(
            name for name in declared if name not in opted_out
        )
        # the exclusions are exactly the annotated ones, not an ambient list
        assert {
            f.name for f in attr.fields(cls) if f.metadata.get("identity") is False
        } == opted_out


class CuratedStagingBackend(RestBackend):
    # differs from CuratedBackend in the resolved base URL and NOTHING else
    config = make_config(base_url="https://staging.example.com")


class CuratedBearerBackend(RestBackend):
    # differs from CuratedBackend in auth kind and NOTHING else; the token is
    # optional, so both connect with an identical (empty) credential profile
    config = make_config(
        auth=AuthConfig(kind="bearer", fields=("token",), optional_fields=("token",))
    )


def test_resolved_base_url_is_identity_bearing() -> None:
    """The bug this pins: a curated profile carries credentials only, so
    repointing base_urls from prod to staging changed no hash -- and cached
    data from the old host was served as current data from the new one."""
    prod = CuratedBackend().connect()
    staging = CuratedStagingBackend().connect()
    assert prod._profile.almost_equals(staging._profile)  # the profile cannot tell
    assert tokenize(prod.read("other")) != tokenize(staging.read("other"))
    assert (
        prod.current_config.content_hash != staging.current_config.content_hash
    )  # ... but the config hash does


def test_auth_kind_is_identity_bearing() -> None:
    none_auth = CuratedBackend().connect()
    bearer = CuratedBearerBackend().connect()
    assert none_auth._profile.almost_equals(bearer._profile)
    assert tokenize(none_auth.read("other")) != tokenize(bearer.read("other"))


def test_path_placeholders_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="not declared params"):
        ResourceConfig(
            name="things",
            schema=things_schema,
            path="/things/{bucket_typo}",  # the typo fails fast, not at fetch
            params=(ParamSpec("owner", required=True),),
        )
    with pytest.raises(ValueError, match="must be required params"):
        ResourceConfig(
            name="things",
            schema=things_schema,
            path="/things/{bucket}",
            params=(ParamSpec("bucket"),),
        )


def test_base_url_key_routes_and_validates() -> None:
    session = FakeSession((FakeResponse([{"id": 1, "name": "a"}]),))
    config = RestBackendConfig(
        base_urls={
            "default": "https://api.example.com",
            "data": "https://data.example.com",
        },
        auth=AuthConfig(kind="none"),
        resources=(
            ResourceConfig(
                name="other", schema=things_schema, path="/other", base_url_key="data"
            ),
        ),
    )
    NativeEngine(session=session).fetch(config, config.get_resource("other"), {}, {})
    ((url, _),) = session.calls
    assert url == "https://data.example.com/other"
    with pytest.raises(ValueError, match="unknown base_urls keys"):
        RestBackendConfig(
            base_urls={"default": "https://api.example.com"},
            auth=AuthConfig(kind="none"),
            resources=(
                ResourceConfig(
                    name="other",
                    schema=things_schema,
                    path="/other",
                    base_url_key="data",
                ),
            ),
        )


def test_paginator_kwargs_accepts_dicts_and_sorts() -> None:
    # a dict must survive direct construction (tuple(dict) keeps only keys)
    by_dict = ResourceConfig(
        name="things",
        schema=things_schema,
        paginator="offset",
        paginator_kwargs={"offset_key": "skip", "limit": 2},
    )
    assert by_dict.paginator_kwargs == (("limit", 2), ("offset_key", "skip"))
    # declaration order is not identity-bearing
    reordered = ResourceConfig(
        name="things",
        schema=things_schema,
        paginator="offset",
        paginator_kwargs=(("offset_key", "skip"), ("limit", 2)),
    )
    assert by_dict == reordered
    assert by_dict.content_hash == reordered.content_hash


def test_schema_to_nullable_dtypes() -> None:
    assert schema_to_nullable_dtypes(things_schema) == {
        "id": "Int64",
        "name": "string",
        "properties": "string",
    }


class FakeResponse:
    def __init__(
        self,
        records: list,
        links: dict | None = None,
        body: dict | None = None,
        status_code: int = 200,
        headers: dict | None = None,
    ) -> None:
        self._records = records
        self.links = links or {}
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> object:
        return self._body if self._body is not None else self._records

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


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
    paginator = OffsetPaginator(limit=2, stop_on_short_page=True)
    params = paginator.initial_params({})
    assert params == {"limit": 2, "offset": 0}
    full = [{"id": 1}, {"id": 2}]
    assert paginator.next(FakeResponse(full), full, "u", params) == (
        "u",
        {"limit": 2, "offset": 2},
    )
    short = [{"id": 3}]
    assert paginator.next(FakeResponse(short), short, "u", params) is None


def test_offset_paginator_requires_a_declared_termination_check() -> None:
    """ "Server clamped my page size" and "data exhausted" are indistinguishable
    on the wire, so the config must say which cross-check applies rather than
    inheriting a silent truncation."""
    with pytest.raises(ValueError, match="declared termination check"):
        OffsetPaginator(limit=100)
    # each of the three declarations is sufficient
    OffsetPaginator(limit=100, total_path="total")
    OffsetPaginator(limit=100, maximum_offset=1_000)
    OffsetPaginator(limit=100, stop_on_short_page=True)


def test_offset_paginator_walks_a_clamping_server_to_exhaustion() -> None:
    """The reproduction: 150 rows on the server, limit=100, server clamps to
    50. The old short-page rule stopped after one request with 50 rows."""
    paginator = OffsetPaginator(limit=100, total_path="total")
    params = paginator.initial_params({})
    assert params == {"limit": 100, "offset": 0}
    clamped = tuple({"id": i} for i in range(50))
    body = {"total": 150}
    # offsets advance by rows RETURNED, not by the requested limit
    assert paginator.next(FakeResponse([], body=body), clamped, "u", params) == (
        "u",
        {"limit": 100, "offset": 50},
    )
    at_100 = {"limit": 100, "offset": 100}
    assert paginator.next(FakeResponse([], body=body), clamped, "u", at_100) is None
    # a missing total is not zero
    with pytest.raises(ValueError, match="no total there"):
        paginator.next(FakeResponse([], body={}), clamped, "u", params)


def test_offset_paginator_maximum_offset_bounds_the_walk() -> None:
    paginator = OffsetPaginator(limit=50, maximum_offset=100)
    full = tuple({"id": i} for i in range(50))
    assert paginator.next(
        FakeResponse(full), full, "u", {"offset": 0, "limit": 50}
    ) == (
        "u",
        {"offset": 50, "limit": 50},
    )
    assert (
        paginator.next(FakeResponse(full), full, "u", {"offset": 50, "limit": 50})
        is None
    )


def test_page_number_paginator() -> None:
    paginator = PageNumberPaginator()
    params = paginator.initial_params({})
    assert params == {"page": 1}
    assert paginator.next(FakeResponse([{}]), [{}], "u", params) == ("u", {"page": 2})
    assert paginator.next(FakeResponse([]), [], "u", params) is None


class FakeSession:
    """Serves canned responses; records (url, params) per request."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list = []

    def get(
        self, url: str, params: dict | None = None, **kwargs: object
    ) -> FakeResponse:
        self.calls.append((url, dict(params or {})))
        return self._responses.pop(0)


class LoopingSession:
    """A server that ignores pagination params and re-serves one page forever."""

    def __init__(self, body: dict) -> None:
        self._body = body
        self.calls: list = []

    def get(
        self, url: str, params: dict | None = None, **kwargs: object
    ) -> FakeResponse:
        self.calls.append((url, dict(params or {})))
        return FakeResponse([], body=self._body)


def make_offset_config(**paginator_kwargs: object) -> RestBackendConfig:
    return RestBackendConfig(
        base_urls={"default": "https://api.example.com"},
        auth=AuthConfig(kind="none"),
        resources=(
            ResourceConfig(
                name="rows",
                schema=things_schema,
                path="/rows",
                record_path="items",
                paginator="offset",
                paginator_kwargs=paginator_kwargs,
            ),
        ),
    )


def test_engine_fetches_a_clamping_server_completely() -> None:
    """150 rows on the server, limit=100, server clamps every page to 50.
    Before the declared cross-check this returned 50 rows from one request and
    cached the truncated frame as complete."""
    config = make_offset_config(limit=100, total_path="total")
    pages = tuple(
        FakeResponse(
            [],
            body={
                "items": [{"id": i, "name": str(i)} for i in range(start, start + 50)],
                "total": 150,
            },
        )
        for start in (0, 50, 100)
    )
    session = FakeSession(pages)
    df = NativeEngine(session=session).fetch(
        config, config.get_resource("rows"), {}, {}
    )
    assert len(df) == 150  # not the clamped 50
    assert len(session.calls) == 3
    assert [params["offset"] for _, params in session.calls] == [0, 50, 100]


def test_engine_bounds_a_server_that_ignores_offset() -> None:
    config = make_offset_config(limit=2, total_path="total")
    session = LoopingSession(
        {"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], "total": 1_000}
    )
    engine = NativeEngine(session=session, max_pages=3)
    with pytest.raises(com.XorqError, match="did not terminate within max_pages=3"):
        engine.fetch(config, config.get_resource("rows"), {}, {})
    assert len(session.calls) == 3  # bounded, not forever


def test_engine_retries_github_style_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("xorq.backends.rest.engines.time.sleep", lambda _: None)
    session = FakeSession(
        (
            FakeResponse(
                [],
                status_code=403,
                headers={"X-RateLimit-Remaining": "0", "Retry-After": "1"},
            ),
            FakeResponse([{"id": 1, "name": "a"}]),
        )
    )
    engine = NativeEngine(session=session)
    config = make_config()
    df = engine.fetch(config, config.get_resource("other"), {}, {})
    assert len(session.calls) == 2  # one rate-limited attempt, one success
    assert df.id.tolist() == [1]


def test_retry_after_parses_both_legal_forms_and_caps() -> None:
    """`Retry-After` is delay-seconds OR an HTTP-date (RFC 9110). A bare
    `int(...)` raised ValueError on the date form, turning the server's
    politeness into a crash mid-fetch. And an uncapped honour of
    `Retry-After: 86400` blocks the pipeline for a day."""
    assert retry_after_seconds(FakeResponse([], headers={"Retry-After": "5"}), 99) == 5
    when = datetime.now(timezone.utc) + timedelta(seconds=10)
    from_date = retry_after_seconds(
        FakeResponse([], headers={"Retry-After": email.utils.format_datetime(when)}), 99
    )
    assert 0 < from_date <= 11
    # a day-long wait is capped, not honoured
    assert (
        retry_after_seconds(FakeResponse([], headers={"Retry-After": "86400"}), 1)
        == MAX_RETRY_WAIT
    )
    # a date in the past is a zero wait, never negative
    past = email.utils.format_datetime(datetime.now(timezone.utc) - timedelta(hours=1))
    assert retry_after_seconds(FakeResponse([], headers={"Retry-After": past}), 99) == 0
    # unparsable and absent both fall back to the caller's backoff, capped
    assert (
        retry_after_seconds(FakeResponse([], headers={"Retry-After": "soon"}), 3) == 3
    )
    assert retry_after_seconds(FakeResponse([]), 3) == 3
    assert retry_after_seconds(FakeResponse([]), 10_000) == MAX_RETRY_WAIT


def test_exhausted_rate_limit_retries_raise_a_diagnosable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old loop called raise_for_status() on the final attempt, so the
    trailing `raise RetryError` was dead code and the operator saw a bare 429
    instead of "you spent the whole retry budget on rate limits"."""
    monkeypatch.setattr("xorq.backends.rest.engines.time.sleep", lambda _: None)
    limited = tuple(
        FakeResponse([], status_code=429, headers={"Retry-After": "1"})
        for _ in range(3)
    )
    session = FakeSession(limited)
    config = make_config()
    engine = NativeEngine(session=session, max_tries=3)
    with pytest.raises(requests.exceptions.RetryError, match="all 3 attempts"):
        engine.fetch(config, config.get_resource("other"), {}, {})
    assert len(session.calls) == 3


def make_enveloped_config(record_path: str = "items") -> RestBackendConfig:
    return RestBackendConfig(
        base_urls={"default": "https://api.example.com"},
        auth=AuthConfig(kind="none"),
        resources=(
            ResourceConfig(
                name="enveloped",
                schema=things_schema,
                path="/enveloped",
                record_path=record_path,
            ),
        ),
    )


def test_missing_record_path_raises_rather_than_reading_as_empty() -> None:
    """ "Path absent from the envelope" and "path present and empty" are
    distinguishable on the wire, so they must not collapse into zero rows: a
    typo'd record_path, or a renamed envelope, otherwise cached a permanently
    empty relation as a complete fetch."""
    config = make_enveloped_config(record_path="data.rows")  # not what the API sends
    body = {"items": [{"id": 1, "name": "a"}], "total": 1}
    with pytest.raises(ValueError, match=r"record_path 'data.rows' is absent"):
        NativeEngine(session=FakeSession((FakeResponse([], body=body),))).fetch(
            config, config.get_resource("enveloped"), {}, {}
        )
    # the message must be debuggable: resource name and the shape that arrived
    with pytest.raises(ValueError, match=r"'enveloped'.*envelope keys: \['items'"):
        NativeEngine(session=FakeSession((FakeResponse([], body=body),))).fetch(
            config, config.get_resource("enveloped"), {}, {}
        )
    # a null at the path is equally not an empty result
    present = make_enveloped_config()
    with pytest.raises(ValueError, match="is null in"):
        NativeEngine(
            session=FakeSession((FakeResponse([], body={"items": None}),))
        ).fetch(present, present.get_resource("enveloped"), {}, {})


def test_present_but_empty_record_path_is_an_empty_page() -> None:
    config = make_enveloped_config()
    df = NativeEngine(
        session=FakeSession((FakeResponse([], body={"items": [], "total": 0}),))
    ).fetch(config, config.get_resource("enveloped"), {}, {})
    assert df.empty
    assert str(df.id.dtype) == "Int64"


def test_make_paginator_unknown() -> None:
    with pytest.raises(ValueError, match="unknown paginator"):
        make_paginator("nope")


def test_absent_column_warns_but_a_sparse_one_stays_quiet() -> None:
    """`reindex`+`record.get` turn a renamed or removed API field into an
    all-NULL column with no signal. Absent from EVERY record on a page is
    distinguishable from per-record nulls, so only that case is reported --
    a genuinely sparse field must not cry wolf."""
    resource = ResourceConfig(name="things", schema=things_schema)
    vanished = ({"id": 1, "properties": "{}"}, {"id": 2, "properties": "{}"})
    with pytest.warns(UserWarning, match=r"'name'.* none of the 2 records"):
        warn_on_absent_columns(vanished, resource)
    # still a truthful answer, just now announced: shaping fills the absent
    # field with NULL exactly as the fetch loop does
    frame = pd.DataFrame(
        [record_to_row(record, resource) for record in vanished]
    ).reindex(columns=tuple(resource.schema))
    assert frame.name.isna().all()
    sparse = ({"id": 1, "name": "a", "properties": "{}"}, {"id": 2, "properties": "{}"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_absent_columns(sparse, resource)
    # an empty page has no columns to speak about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_absent_columns((), resource)
    # a declared residual_column is not an API field, so it is never reported
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_absent_columns(
            ({"id": 1, "name": "a"},),
            ResourceConfig(
                name="things", schema=things_schema, residual_column="properties"
            ),
        )


def test_record_to_row_nests_to_json() -> None:
    resource = ResourceConfig(
        name="things", schema=things_schema, residual_column="properties"
    )
    row = record_to_row({"id": 1, "name": {"nested": True}, "extra": "x"}, resource)
    assert row["id"] == 1
    assert json.loads(row["name"]) == {"nested": True}
    # the declared overflow column: only unmapped fields, no duplication
    props = json.loads(row["properties"])
    assert props == {"extra": "x"}
    assert "id" not in props and "name" not in props


def test_residual_column_is_declared_not_name_sniffed() -> None:
    # without a declaration, a field literally named "properties" is just a
    # typed column -- the record's own value lands there
    resource = ResourceConfig(name="things", schema=things_schema)
    row = record_to_row({"id": 1, "properties": {"real": True}}, resource)
    assert json.loads(row["properties"]) == {"real": True}
    with pytest.raises(ValueError, match="not a schema column"):
        ResourceConfig(name="things", schema=things_schema, residual_column="nope")


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


def _apply_query_key(auth: AuthConfig, credentials: dict) -> dict:
    # a "params" key rides the query string (query-param API keys)
    return {"params": {"api_key": credentials.get(auth.fields[0])}}


def make_query_key_config() -> RestBackendConfig:
    return RestBackendConfig(
        base_urls={"default": "https://api.example.com"},
        auth=AuthConfig(kind="query_key", fields=("key",)),
        resources=(ResourceConfig(name="other", schema=things_schema, path="/other"),),
    )


def test_custom_auth_kind_via_applier_registry() -> None:
    session = FakeSession((FakeResponse([{"id": 1, "name": "a"}]),))
    engine = NativeEngine(
        auth_appliers={**RestBackend.auth_appliers, "query_key": _apply_query_key},
        session=session,
    )
    config = make_query_key_config()
    engine.fetch(config, config.get_resource("other"), {}, {"key": "sekret"})
    ((_, params),) = session.calls
    assert params == {"api_key": "sekret"}


def test_unknown_auth_kind_rejected_at_connect() -> None:
    class NoApplierBackend(RestBackend):
        config = make_query_key_config()

    with pytest.raises(com.XorqError, match="no auth applier for kind 'query_key'"):
        NoApplierBackend().connect(key="k")


def test_table_schema_arg_rejected_for_resources() -> None:
    con = CuratedBackend().connect()
    with pytest.raises(com.XorqError, match="declared in the config"):
        con.table("things", schema=things_schema, bucket="b")


def test_init_subclass_rejects_misnamed_var_kwargs_bucket() -> None:
    with pytest.raises(TypeError, match=r"must be named \*\*kwargs"):

        class MisnamedBucketBackend(RestBackend):
            def do_connect(self, *, config: dict | None = None, **creds: str) -> None:
                pass


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
