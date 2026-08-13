from __future__ import annotations

import pytest

from xorq.common.utils.func_utils import log_excepts, maybe_log_excepts


# `log_excepts` decorates the four Flight RPC handlers in flight/server.py, but
# only when `options.debug` is set -- which CI never sets. So nothing else in
# the suite covers it, and the swallow it used to do was invisible until it hit
# someone: under XORQ_DEBUG=1 alone, a failed exchange was caught at the RPC
# boundary and returned None, i.e. a clean, empty, cacheable stream, undoing
# the propagation the exchangers had just been fixed to guarantee. These tests
# pin the decorator to "adds logging, changes nothing else".


def boom() -> None:
    raise ValueError("boom")


def test_log_excepts_propagates_the_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        log_excepts(boom)()


def test_log_excepts_propagates_the_same_instance() -> None:
    """Not a re-wrap: the caller sees the original error, traceback included."""
    error = ValueError("boom")

    def raise_error() -> None:
        raise error

    with pytest.raises(ValueError) as excinfo:
        log_excepts(raise_error)()
    assert excinfo.value is error


def test_log_excepts_returns_the_value_when_nothing_raises() -> None:
    assert log_excepts(lambda: 42)() == 42


def test_log_excepts_does_not_catch_a_narrower_exception_arg() -> None:
    """`exception` selects what is *logged*; nothing is swallowed either way."""
    with pytest.raises(ValueError, match="boom"):
        log_excepts(boom, exception=KeyError)()


@pytest.mark.parametrize(
    "debug",
    (
        pytest.param(True, id="debug-on"),
        pytest.param(False, id="debug-off"),
    ),
)
def test_maybe_log_excepts_propagates_whether_or_not_debug_is_on(
    debug: bool,
) -> None:
    """The point of the whole thing: debug mode must not change semantics.

    With debug off `maybe_log_excepts` hands back the undecorated function, so
    these two paths agreeing is what keeps a failure from depending on an env
    var to reach the caller.
    """
    with pytest.raises(ValueError, match="boom"):
        maybe_log_excepts(boom, debug=debug)()
