from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from xorq.common.utils.func_utils import log_excepts, maybe_log_excepts


# `log_excepts` decorates the four Flight RPC handlers only when `options.debug`
# is set, which CI never sets -- so nothing else covers it, and its swallow
# silently undid the exchangers' propagation under XORQ_DEBUG=1 alone. These
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
    """Debug mode must not change semantics.

    With debug off the function comes back undecorated, so these two agreeing
    is what keeps a failure from depending on an env var to reach the caller.
    """
    with pytest.raises(ValueError, match="boom"):
        maybe_log_excepts(boom, debug=debug)()


def test_log_excepts_logs_an_exception_raised_while_a_generator_runs() -> None:
    """A generator's body runs on iteration, after the call has returned.

    Flight's ``do_action`` is a generator, so wrapping only the call logged
    ``exiting`` before any work and never saw an error raised in the body.
    """

    def gen():
        yield 1
        raise ValueError("boom")

    with capture_logs() as logs:
        wrapped = log_excepts(gen)()
        assert [event["event"] for event in logs] == []
        with pytest.raises(ValueError, match="boom"):
            list(wrapped)

    assert [event["log_level"] for event in logs][-1] == "error"
    assert not any("exiting" in event["event"] for event in logs)


def test_log_excepts_logs_exiting_after_a_generator_is_exhausted() -> None:
    def gen():
        yield from (1, 2)

    with capture_logs() as logs:
        assert list(log_excepts(gen)()) == [1, 2]

    assert "exiting" in logs[-1]["event"]
