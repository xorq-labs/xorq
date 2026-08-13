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

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)


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
        self, resp: requests.Response, records: tuple, url: str, params: dict
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
        self, resp: requests.Response, records: tuple, url: str, params: dict
    ) -> tuple[str, dict] | None:
        return None


@frozen
class HeaderLinkPaginator(BasePaginator):
    """RFC 5988 ``Link: <...>; rel="next"`` (GitHub, GitLab, ...). The next
    URL carries its own query string, so params are dropped."""

    rel = field(validator=instance_of(str), default="next")

    def next(
        self, resp: requests.Response, records: tuple, url: str, params: dict
    ) -> tuple[str, dict] | None:
        next_url = resp.links.get(self.rel, {}).get("url")
        return (next_url, {}) if next_url else None


@frozen
class JsonLinkPaginator(BasePaginator):
    """Next-page URL embedded in the response body at ``path`` (dotted)."""

    path = field(validator=instance_of(str), default="next")

    def next(
        self, resp: requests.Response, records: tuple, url: str, params: dict
    ) -> tuple[str, dict] | None:
        next_url = toolz.get_in(self.path.split("."), resp.json())
        return (next_url, {}) if next_url else None


_MISSING = object()


@frozen
class OffsetPaginator(BasePaginator):
    """``offset``/``limit`` query params, with a declared termination check.

    Terminating on a short page alone is unsafe, and not fixable by
    representation: "the server clamped my page size" and "the data is
    exhausted" are genuinely indistinguishable on the wire. Against an API
    that clamps -- ask for 100, get 50, very common -- *every* full page
    satisfies ``len(records) < limit``, so the fetch stops after one request
    and a truncated frame caches as a complete one. So the config must declare
    which cross-check applies (dlt draws the same line):

    - ``total_path``: dotted path to the total record count in the response
      body. The strongest check -- pagination runs until the offset reaches it,
      and a clamping server is simply walked in smaller steps. It also governs
      the empty-page case: an empty page below the declared total is a hole, not
      the end, and raises rather than truncating (see
      ``_maybe_raise_on_sparse_page``).
    - ``maximum_offset``: a hard offset bound, for APIs that publish no total.
    - ``stop_on_short_page=True``: an explicit acknowledgement that for this
      API a short page does mean exhaustion. Declaring it is how an author
      consciously accepts the clamp risk instead of inheriting it silently.

    Offsets advance by the number of records actually returned, never by the
    requested limit, so a clamped page does not skip rows. The unconditional
    page bound against a server that ignores ``offset`` altogether lives in the
    engine (``NativeEngine.max_pages``), where it covers every paginator.
    """

    limit = field(validator=instance_of(int), default=100)
    offset_key = field(validator=instance_of(str), default="offset")
    limit_key = field(validator=instance_of(str), default="limit")
    total_path = field(validator=optional(instance_of(str)), default=None)
    maximum_offset = field(validator=optional(instance_of(int)), default=None)
    stop_on_short_page = field(validator=instance_of(bool), default=False)

    def __attrs_post_init__(self) -> None:
        if (
            self.total_path is None
            and self.maximum_offset is None
            and not self.stop_on_short_page
        ):
            raise ValueError(
                "offset pagination needs a declared termination check: "
                "total_path=<dotted path to a total count>, "
                "maximum_offset=<int>, or stop_on_short_page=True to accept "
                "that a short page means exhaustion for this API. Without one, "
                "a server that clamps the page size truncates the result "
                "silently and the partial fetch caches as complete."
            )

    def initial_params(self, params: dict) -> dict:
        return {**params, self.limit_key: self.limit, self.offset_key: 0}

    def next(
        self, resp: requests.Response, records: tuple, url: str, params: dict
    ) -> tuple[str, dict] | None:
        if not records:
            # An empty page is the end -- unless a declared total says
            # otherwise, in which case it is a hole and stopping here would
            # truncate. See `_maybe_raise_on_sparse_page`.
            self._maybe_raise_on_sparse_page(resp, params)
            return None
        offset = params[self.offset_key] + len(records)
        if self.total_path is not None:
            if offset >= self._total(resp):
                return None
        elif self.maximum_offset is not None:
            if offset >= self.maximum_offset:
                return None
        elif len(records) < self.limit:
            return None
        return (url, {**params, self.offset_key: offset})

    def _maybe_raise_on_sparse_page(
        self, resp: requests.Response, params: dict
    ) -> None:
        """Refuse to terminate on an empty page the declared total contradicts.

        Terminating on any empty page silently defeated the cross-check this
        paginator forces a config to declare: a config carrying ``total_path``
        still truncated on a sparse mid-stream page -- one hole and the fetch
        stopped, with the partial result cached as complete. That is the exact
        failure ``total_path`` exists to make impossible, so where the total
        says rows remain, this raises rather than returning fewer rows than the
        server said it had. Consistent with a missing total, which already
        raises: a declared cross-check that cannot be satisfied is an error, not
        a default.

        ``maximum_offset`` and ``stop_on_short_page`` are deliberately NOT
        cross-checked here. Neither claims to know how many records exist, so an
        empty page is the only end-of-data signal they have; only a declared
        total can contradict one.
        """
        if self.total_path is None:
            return
        offset = params[self.offset_key]
        total = self._total(resp)
        if offset < total:
            raise ValueError(
                f"offset pagination: empty page at {self.offset_key}={offset}, "
                f"but the declared total_path {self.total_path!r} says the "
                f"server holds {total} records. Stopping here would return a "
                "truncated result and cache the partial fetch as complete -- "
                "which is what declaring a cross-check is meant to prevent. If "
                "this API genuinely serves holes mid-stream, its total is not a "
                "usable termination check: bound the walk with "
                "maximum_offset=<int> instead, or declare "
                "stop_on_short_page=True if a short page really does mean "
                "exhaustion for it."
            )

    def _total(self, resp: requests.Response) -> int:
        total = toolz.get_in(self.total_path.split("."), resp.json(), _MISSING)
        if total is _MISSING or total is None:
            raise ValueError(
                f"offset pagination declares total_path {self.total_path!r} but "
                "the response carries no total there; absence is not zero -- "
                "check the path against the API's current response shape"
            )
        return int(total)


@frozen
class PageNumberPaginator(BasePaginator):
    """``page=N`` query param starting at ``start``; stops on an empty page.

    An empty page is a genuine terminator here (unlike a *full* page under
    offset pagination, which a clamping server makes ambiguous), so no
    cross-check is required. The residual hazard is an API that re-serves the
    last page for an out-of-range page number and so never returns an empty
    one; that is bounded by ``NativeEngine.max_pages``, which raises rather
    than looping forever.
    """

    page_key = field(validator=instance_of(str), default="page")
    start = field(validator=instance_of(int), default=1)

    def initial_params(self, params: dict) -> dict:
        return {**params, self.page_key: self.start}

    def next(
        self, resp: requests.Response, records: tuple, url: str, params: dict
    ) -> tuple[str, dict] | None:
        if not records:
            return None
        return (url, {**params, self.page_key: params[self.page_key] + 1})


# Append-only: names are declarative config values and thus identity-bearing.
# Backends extend by *merging over* this base registry (a class-level
# `paginators` mapping picked up by `RestBackend.make_engine`); namespace
# custom names (e.g. "mixpanel.session_id") to keep the base set unambiguous.
PAGINATORS = MappingProxyType(
    {
        "single_page": SinglePagePaginator,
        "header_link": HeaderLinkPaginator,
        "json_link": JsonLinkPaginator,
        "offset": OffsetPaginator,
        "page_number": PageNumberPaginator,
    }
)


def make_paginator(
    name: str | None,
    paginator_kwargs: tuple = (),
    registry: MappingProxyType = PAGINATORS,
) -> Paginator:
    if name is None:
        name = "single_page"
    try:
        cls = registry[name]
    except KeyError:
        raise ValueError(
            f"unknown paginator {name!r}; available: {sorted(registry)}"
        ) from None
    return cls(**dict(paginator_kwargs))
