"""Extraction engines behind the REST contract
(ADR-2215).

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

import email.utils
import itertools
import json
import time
import warnings
from datetime import (
    datetime,
    timezone,
)
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Protocol

import pandas as pd
import requests
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import instance_of

import xorq.common.exceptions as com
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


# A server can legally ask us to wait a day (`Retry-After: 86400`). Honouring
# that would hang a pipeline with no output; cap the wait and let the retry
# budget run out with a diagnosable error instead.
MAX_RETRY_WAIT = 60.0


def retry_after_seconds(
    resp: requests.Response, default: float, cap: float = MAX_RETRY_WAIT
) -> float:
    """Seconds to wait per RFC 9110 ``Retry-After``, capped.

    The header has two legal forms: delay-seconds and an HTTP-date. A bare
    ``int(...)`` raises ValueError on the date form -- turning a server's
    politeness into a crash mid-fetch -- so parse both, and fall back to the
    caller's backoff for anything unparsable.
    """
    value = resp.headers.get("Retry-After") if resp.headers else None
    if value is None:
        return min(default, cap)
    value = str(value).strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return min(default, cap)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return min(max(seconds, 0.0), cap)


# -- record shaping ----------------------------------------------------------


_MISSING = object()


def extract_records(resp: requests.Response, resource_config: ResourceConfig) -> tuple:
    """The records on one page, at the resource's declared ``record_path``.

    A missing path RAISES. Defaulting it to ``()`` collapsed two facts that
    the wire distinguishes perfectly well -- "the envelope has no such key"
    (a typo'd `record_path`, or an API that renamed its envelope) and "the key
    is there and empty" -- into zero rows with no error, on every page. With
    page-number pagination the first "empty page" then terminates the fetch, so
    a permanently empty relation cached as a complete one. A genuinely present
    empty list still yields an empty page, which is the honest answer.
    """
    data = resp.json()
    if resource_config.record_path:
        keys = resource_config.record_path.split(".")
        found = toolz.get_in(keys, data, _MISSING)
        if found is _MISSING or found is None:
            raise ValueError(
                f"resource {resource_config.name!r}: record_path "
                f"{resource_config.record_path!r} is "
                + ("null in" if found is None else "absent from")
                + f" the response envelope from {getattr(resp, 'url', None)!r}"
                + f" (envelope keys: {_envelope_keys(data)}). An empty result "
                "must be an empty list at that path; check the path against the "
                "API's current response shape."
            )
        data = found
    if isinstance(data, dict):
        # a single-record resource (e.g. /repos/{owner}/{repo})
        return (data,)
    return tuple(data)


def _envelope_keys(data: object) -> object:
    """The shape of what came back, for a record_path error message."""
    if isinstance(data, dict):
        return sorted(data)
    return f"<{type(data).__name__}>"


def warn_on_absent_columns(records: tuple, resource_config: ResourceConfig) -> None:
    """Warn when a declared column is absent from EVERY record on a page.

    Shaping fills a missing field with NULL, so an API field that was renamed
    or removed becomes an all-NULL column with no signal at all. The
    distinction that makes a signal possible: a genuinely sparse field is
    absent from *some* records, a vanished field from *all* of them. So only
    the all-of-them case is reported, and a sparse field stays quiet.

    A warning, not an error: pages legitimately differ (an API can omit a whole
    block for a page whose records are all of one kind), and raising would turn
    an upstream field rename into a hard pipeline failure. NULLs are still a
    truthful answer -- what was missing was the notification. Extra fields the
    API sends are not reported: capturing them is what ``residual_column``
    declares, and every unmapped field on a wide API would be noise.
    """
    if not records:
        return
    named = tuple(
        name
        for name in resource_config.schema.names
        if name != resource_config.residual_column
    )
    absent = tuple(
        name
        for name in named
        if not any(isinstance(record, dict) and name in record for record in records)
    )
    if absent:
        warnings.warn(
            f"resource {resource_config.name!r}: declared column(s) "
            f"{absent} appear in none of the {len(records)} records on this "
            "page, so they will be all-NULL. An API field that was renamed or "
            "removed looks exactly like this; a field that is merely sparse "
            "does not (it is present in some records).",
            UserWarning,
            stacklevel=3,
        )


def frame_from_records(records: tuple, resource_config: ResourceConfig) -> pd.DataFrame:
    """One page of records as a schema-conformed frame: conforming per page
    (not once at the end) is what makes pages independently consumable."""
    warn_on_absent_columns(records, resource_config)
    return (
        pd.DataFrame([record_to_row(record, resource_config) for record in records])
        .reindex(columns=tuple(resource_config.schema))
        .astype(resource_config.dtypes)
    )


def record_to_row(record: dict, resource_config: ResourceConfig) -> dict:
    def render(value: object) -> object:
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return value

    residual_column = resource_config.residual_column
    named = tuple(n for n in resource_config.schema.names if n != residual_column)
    row = {name: render(record.get(name)) for name in named}
    if residual_column is not None:
        # the overflow column: only fields not already given a typed
        # column, so the residual doesn't duplicate them at catalog scale
        residual = {k: v for k, v in record.items() if k not in named}
        row[residual_column] = json.dumps(residual, sort_keys=True)
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
    # unconditional page bound, across every paginator: a server that ignores
    # `offset`, or re-serves its last page for an out-of-range `page`, would
    # otherwise loop forever. Transport safety, so it carries no identity.
    max_pages = field(validator=instance_of(int), default=10_000)

    def fetch(
        self,
        config: RestBackendConfig,
        resource_config: ResourceConfig,
        params: dict,
        credentials: Mapping,
    ) -> pd.DataFrame:
        return pd.concat(
            self.fetch_batches(config, resource_config, params, credentials),
            ignore_index=True,
        )

    def fetch_batches(
        self,
        config: RestBackendConfig,
        resource_config: ResourceConfig,
        params: dict,
        credentials: Mapping,
    ) -> Iterator[pd.DataFrame]:
        """Page-wise fetch: one schema-conformed frame per HTTP page.

        The intermediary between pagination and materialization: ``fetch``
        concatenates, and a future streaming consumer (spill-to-parquet at
        the Read boundary) can pull pages without holding the whole result.
        Deliberately not part of the :class:`Engine` protocol until such a
        consumer exists — streaming is transport, never identity.

        Always yields at least one (possibly empty) frame, so ``pd.concat``
        needs no empty-input guard and dtypes hold for empty results.
        """
        paginator = make_paginator(
            resource_config.paginator,
            resource_config.paginator_kwargs,
            registry=self.paginators,
        )
        url = config.base_url(
            resource_config.base_url_key
        ) + resource_config.path.format(**params)
        # scope params are identity-only discriminators
        # (ADR-build-artifacts-are-credential-free): they must not reach the
        # wire, where a server would ignore them at best
        on_the_wire_excluded = set(resource_config.path_placeholders) | set(
            resource_config.scope_params
        )
        query = {k: v for k, v in params.items() if k not in on_the_wire_excluded}
        request_kwargs = dict(self._auth_kwargs(config.auth, credentials))
        query = {**query, **request_kwargs.pop("params", {})}
        query = paginator.initial_params(query)
        for page in itertools.count(1):
            resp = self._get_with_backoff(url, query, request_kwargs)
            records = extract_records(resp, resource_config)
            yield frame_from_records(records, resource_config)
            nxt = paginator.next(resp, records, url, query)
            if nxt is None:
                return
            if page >= self.max_pages:
                raise com.XorqError(
                    f"resource {resource_config.name!r}: pagination did not "
                    f"terminate within max_pages={self.max_pages} (last request "
                    f"{url} with {query}). A server that ignores the pagination "
                    "param -- or re-serves its last page for an out-of-range "
                    "page number -- looks like this; check the paginator config, "
                    "or raise the engine's max_pages if the resource really is "
                    "this large."
                )
            url, query = nxt

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
        """GET with rate-limit backoff.

        The exhausted-budget path is reachable and says what happened: the
        previous shape called ``raise_for_status()`` on the final attempt, so
        the 429 surfaced as a bare HTTPError and the "exceeded N tries" message
        was dead code. Transient 5xx and connection errors are deliberately
        still NOT retried -- only rate limits are -- which is worth revisiting
        but is a behavior change of its own.
        """
        for tries in range(1, self.max_tries + 1):
            resp = self.session.get(
                url, params=params, timeout=self.timeout, **request_kwargs
            )
            if not _is_rate_limited(resp):
                resp.raise_for_status()
                return resp
            if tries == self.max_tries:
                break
            time.sleep(retry_after_seconds(resp, default=2**tries))
        raise requests.exceptions.RetryError(
            f"rate limited by {url} on all {self.max_tries} attempts "
            f"(last status {resp.status_code}, "
            f"Retry-After={(resp.headers or {}).get('Retry-After')!r})"
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
