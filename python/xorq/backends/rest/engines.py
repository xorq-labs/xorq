"""Extraction engines behind the REST contract (ADR-0018).

The config (``config.py``) is the stable contract; an :class:`Engine` is a
swappable executor of that contract. ``NativeEngine`` (paginator-driven,
dependency-free) is the default; a dlt-backed engine is a future alternative
behind the same interface. The engine-equivalence obligation: same config,
any engine, same rows.

Engines are constructed with their extension registries (paginators, auth
appliers) so backends extend by *declaring* — a class-level mapping merged
over the base registry — never by overriding private fetch methods.
"""

from __future__ import annotations

import json
import string
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Protocol

import pandas as pd
import requests
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import instance_of

from xorq.backends.rest.paginators import (
    PAGINATORS,
    make_paginator,
)


if TYPE_CHECKING:
    from xorq.backends.rest.config import (
        AuthConfig,
        ResourceConfig,
        RestBackendConfig,
    )


class Engine(Protocol):
    """Structural contract for an extraction engine.

    ``fetch`` executes one resource read: given the API config, the resource
    config, the read-time params, and the (already env-resolved) credentials,
    return a DataFrame conforming to the resource's schema. Engines must be
    interchangeable: same config, any engine, same rows.
    """

    def fetch(
        self,
        config: RestBackendConfig,
        resource_config: ResourceConfig,
        params: dict,
        credentials: Mapping,
    ) -> pd.DataFrame: ...


# -- auth appliers -----------------------------------------------------------
#
# An applier maps (AuthConfig, credentials) -> requests kwargs. A "params"
# key is merged into the query string (e.g. query-param API keys); every
# other key ("headers", "auth", ...) is passed to requests verbatim.


def apply_basic_auth(auth: AuthConfig, credentials: Mapping) -> dict:
    return {
        "auth": (
            credentials.get(auth.username_field, ""),
            credentials.get(auth.password_field, ""),
        )
    }


def apply_bearer_auth(auth: AuthConfig, credentials: Mapping) -> dict:
    token = credentials.get(auth.resolved_token_field)
    return {"headers": {"Authorization": f"Bearer {token}"}} if token else {}


def apply_no_auth(auth: AuthConfig, credentials: Mapping) -> dict:
    return {}


AUTH_APPLIERS: Mapping[str, Callable] = MappingProxyType(
    {
        "basic": apply_basic_auth,
        "bearer": apply_bearer_auth,
        "none": apply_no_auth,
    }
)


def _is_rate_limited(resp: requests.Response) -> bool:
    # 429 is the standard signal; GitHub signals primary rate limits with
    # 403 + X-RateLimit-Remaining: 0
    return resp.status_code == 429 or (
        resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0"
    )


# -- record shaping ----------------------------------------------------------


def extract_records(resp: requests.Response, resource_config: ResourceConfig) -> tuple:
    data = resp.json()
    if resource_config.record_path:
        data = toolz.get_in(resource_config.record_path.split("."), data, ())
    if isinstance(data, dict):
        # a single-record resource (e.g. /repos/{owner}/{repo})
        return (data,)
    return tuple(data)


def record_to_row(record: dict, resource_config: ResourceConfig) -> dict:
    def render(value: object) -> object:
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return value

    named = tuple(n for n in resource_config.schema.names if n != "properties")
    row = {name: render(record.get(name)) for name in named}
    if "properties" in resource_config.schema.names:
        # the overflow column: only fields not already given a typed
        # column, so `properties` doesn't duplicate them at catalog scale
        residual = {k: v for k, v in record.items() if k not in named}
        row["properties"] = json.dumps(residual, sort_keys=True)
    return row


@frozen
class NativeEngine:
    """The default paginator-driven engine: GET, backoff, record shaping.

    Registries arrive at construction (``RestBackend.make_engine`` passes the
    backend's class-level ``paginators`` / ``auth_appliers``), so backend
    extensions flow through without touching this class.
    """

    paginators = field(default=PAGINATORS)
    auth_appliers = field(default=AUTH_APPLIERS)
    # reused across pages for connection keep-alive; carries no identity
    session = field(factory=requests.Session, eq=False)
    max_tries = field(validator=instance_of(int), default=5)
    timeout = field(validator=instance_of(int), default=600)

    def fetch(
        self,
        config: RestBackendConfig,
        resource_config: ResourceConfig,
        params: dict,
        credentials: Mapping,
    ) -> pd.DataFrame:
        paginator = make_paginator(
            resource_config.paginator,
            resource_config.paginator_kwargs,
            registry=self.paginators,
        )
        url = config.base_url() + resource_config.path.format(**params)
        path_params = {
            name
            for _, name, *_ in string.Formatter().parse(resource_config.path)
            if name
        }
        query = {k: v for k, v in params.items() if k not in path_params}
        request_kwargs = dict(self._auth_kwargs(config.auth, credentials))
        query = {**query, **request_kwargs.pop("params", {})}
        query = paginator.initial_params(query)
        rows: list[dict] = []
        while True:
            resp = self._get_with_backoff(url, query, request_kwargs)
            records = extract_records(resp, resource_config)
            rows.extend(record_to_row(record, resource_config) for record in records)
            nxt = paginator.next(resp, records, url, query)
            if nxt is None:
                break
            url, query = nxt
        return (
            pd.DataFrame(rows)
            .reindex(columns=tuple(resource_config.schema))
            .astype(resource_config.dtypes)
        )

    def _auth_kwargs(self, auth: AuthConfig, credentials: Mapping) -> dict:
        try:
            applier = self.auth_appliers[auth.kind]
        except KeyError:
            raise ValueError(
                f"no auth applier for kind {auth.kind!r}; "
                f"available: {sorted(self.auth_appliers)}"
            ) from None
        return applier(auth, credentials)

    def _get_with_backoff(
        self, url: str, params: dict, request_kwargs: dict
    ) -> requests.Response:
        for tries in range(1, self.max_tries + 1):
            resp = self.session.get(
                url, params=params, timeout=self.timeout, **request_kwargs
            )
            if _is_rate_limited(resp) and tries != self.max_tries:
                time.sleep(int(resp.headers.get("Retry-After", 2**tries)))
                continue
            resp.raise_for_status()
            return resp
        raise requests.exceptions.RetryError(
            f"exceeded {self.max_tries} tries for {url}"
        )


@frozen
class FetchOverrideEngine:
    """Adapts a resource's ``fetch_override`` callable to the Engine
    protocol, so override resources are an engine *selection* rather than a
    special case. Overrides receive the backend (execution-time clients live
    there, e.g. mixpanel's), which is why this engine is built per-backend."""

    backend = field(eq=False)

    def fetch(
        self,
        config: RestBackendConfig,
        resource_config: ResourceConfig,
        params: dict,
        credentials: Mapping,
    ) -> pd.DataFrame:
        return resource_config.fetch_override(self.backend, **params)
