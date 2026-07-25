"""Native pagination strategies for the REST contract.

Original implementations of the standard pagination patterns; the strategy
*interface* (advance from the previous response to the next request, or
stop) is distilled from the shape of dlt's ``BasePaginator``, but no dlt
code is copied — these are dependency-free by design (the default engine;
dlt is an optional engine behind the same contract for the long tail).

A paginator answers one question: given the previous response and the
records extracted from it, what is the next ``(url, params)`` — or None to
stop. Registered by name (``PAGINATORS``); ``ResourceConfig.paginator`` is
that name, making the choice declarative and identity-bearing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import instance_of


if TYPE_CHECKING:
    import requests


class Paginator(Protocol):
    """Structural contract for a pagination strategy.

    ``initial_params`` seeds the first request (e.g. offset/page params);
    strategies that don't need it inherit the pass-through default from
    ``BasePaginator``. ``next`` answers: given the previous response and the
    records extracted from it, what is the next ``(url, params)`` — or
    ``None`` to stop. Structural, so the concrete ``@frozen`` strategies
    below satisfy it by shape rather than by inheriting it.
    """

    def initial_params(self, params: dict) -> dict: ...

    def next(
        self, resp: requests.Response, records: list, url: str, params: dict
    ) -> tuple[str, dict] | None: ...


class BasePaginator:
    """Shared implementation base for the native strategies. Supplies the
    pass-through ``initial_params`` so the engine can call it unconditionally
    (no ``getattr`` probe); strategies that seed params override it."""

    def initial_params(self, params: dict) -> dict:
        return params


@frozen
class SinglePagePaginator(BasePaginator):
    """One request, no pagination (the default when paginator is None)."""

    def next(
        self, resp: requests.Response, records: list, url: str, params: dict
    ) -> tuple[str, dict] | None:
        return None


@frozen
class HeaderLinkPaginator(BasePaginator):
    """RFC 5988 ``Link: <...>; rel="next"`` (GitHub, GitLab, ...). The next
    URL carries its own query string, so params are dropped."""

    rel = field(validator=instance_of(str), default="next")

    def next(
        self, resp: requests.Response, records: list, url: str, params: dict
    ) -> tuple[str, dict] | None:
        next_url = resp.links.get(self.rel, {}).get("url")
        return (next_url, {}) if next_url else None


@frozen
class JsonLinkPaginator(BasePaginator):
    """Next-page URL embedded in the response body at ``path`` (dotted)."""

    path = field(validator=instance_of(str), default="next")

    def next(
        self, resp: requests.Response, records: list, url: str, params: dict
    ) -> tuple[str, dict] | None:
        next_url = toolz.get_in(self.path.split("."), resp.json())
        return (next_url, {}) if next_url else None


@frozen
class OffsetPaginator(BasePaginator):
    """``offset``/``limit`` query params; stops on a short page."""

    limit = field(validator=instance_of(int), default=100)
    offset_key = field(validator=instance_of(str), default="offset")
    limit_key = field(validator=instance_of(str), default="limit")

    def initial_params(self, params: dict) -> dict:
        return {**params, self.limit_key: self.limit, self.offset_key: 0}

    def next(
        self, resp: requests.Response, records: list, url: str, params: dict
    ) -> tuple[str, dict] | None:
        if len(records) < self.limit:
            return None
        return (url, {**params, self.offset_key: params[self.offset_key] + self.limit})


@frozen
class PageNumberPaginator(BasePaginator):
    """``page=N`` query param starting at ``start``; stops on an empty page."""

    page_key = field(validator=instance_of(str), default="page")
    start = field(validator=instance_of(int), default=1)

    def initial_params(self, params: dict) -> dict:
        return {**params, self.page_key: self.start}

    def next(
        self, resp: requests.Response, records: list, url: str, params: dict
    ) -> tuple[str, dict] | None:
        if not records:
            return None
        return (url, {**params, self.page_key: params[self.page_key] + 1})


# Append-only: names are declarative config values and thus identity-bearing.
PAGINATORS = {
    "single_page": SinglePagePaginator,
    "header_link": HeaderLinkPaginator,
    "json_link": JsonLinkPaginator,
    "offset": OffsetPaginator,
    "page_number": PageNumberPaginator,
}


def make_paginator(name: str | None, paginator_kwargs: tuple = ()) -> Paginator:
    if name is None:
        name = "single_page"
    try:
        cls = PAGINATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown paginator {name!r}; available: {sorted(PAGINATORS)}"
        ) from None
    return cls(**dict(paginator_kwargs))
