"""Unit tests for FlightClient._wait_on_healthcheck.

All tests bypass __init__ (which needs a live server) via object.__new__
and patch only self.do_action + time.sleep.

pa.ArrowIOError is OSError in this PyArrow version (direct alias, not subclass).
Generic-exception tests use ValueError to avoid hitting the ArrowIOError handler.
"""

import pyarrow as pa
import pytest

from xorq.flight.client import FlightClient


def _make_client():
    """Return a FlightClient with no live connection."""
    return object.__new__(FlightClient)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_healthcheck_returns_on_first_success(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(return_value=iter([]))
    mock_sleep = mocker.patch("xorq.flight.client.time.sleep")

    client._wait_on_healthcheck(n_tries=3, sleep_n=0)

    client.do_action.assert_called_once()
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# FlightUnauthenticatedError → return immediately (no RuntimeError, no sleep)
# ---------------------------------------------------------------------------


def test_healthcheck_returns_on_unauthenticated_error(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(
        side_effect=pa.flight.FlightUnauthenticatedError("auth required")
    )
    mock_sleep = mocker.patch("xorq.flight.client.time.sleep")

    client._wait_on_healthcheck(n_tries=5, sleep_n=0)

    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# generic Exception → pass → sleep → retry
# ---------------------------------------------------------------------------


def test_healthcheck_swallows_generic_exception_and_sleeps(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(side_effect=ValueError("unexpected"))
    mock_sleep = mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(RuntimeError):
        client._wait_on_healthcheck(n_tries=2, sleep_n=7)

    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(7)


def test_healthcheck_generic_exception_does_not_propagate(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(side_effect=ValueError("connection refused"))
    mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(RuntimeError):
        client._wait_on_healthcheck(n_tries=1, sleep_n=0)


# ---------------------------------------------------------------------------
# pa.ArrowIOError with "Deadline" → log and retry, succeed on next attempt
# ---------------------------------------------------------------------------


def test_healthcheck_retries_on_deadline_arrow_io_error(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(
        side_effect=[
            pa.ArrowIOError("Deadline exceeded"),
            pa.ArrowIOError("Deadline exceeded"),
            iter([]),
        ]
    )
    mock_sleep = mocker.patch("xorq.flight.client.time.sleep")

    client._wait_on_healthcheck(n_tries=5, sleep_n=1)

    assert client.do_action.call_count == 3
    assert mock_sleep.call_count == 2


def test_healthcheck_non_deadline_arrow_io_error_raises(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(side_effect=pa.ArrowIOError("Connection reset"))
    mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(pa.ArrowIOError, match="Connection reset"):
        client._wait_on_healthcheck(n_tries=3, sleep_n=0)


# ---------------------------------------------------------------------------
# FlightUnavailableError with SSL/Socket match → raise e
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "Ssl handshake failed: SSL_ERROR_SSL: error:0A000086:SSL routines::certificate verify failed",
        "Socket closed",
    ],
)
def test_healthcheck_raises_on_ssl_or_socket_closed(mocker, msg):
    client = _make_client()
    client.do_action = mocker.MagicMock(
        side_effect=pa.flight.FlightUnavailableError(msg)
    )
    mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(pa.flight.FlightUnavailableError, match=msg[:20]):
        client._wait_on_healthcheck(n_tries=3, sleep_n=0)


def test_healthcheck_swallows_non_matching_unavailable_error(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(
        side_effect=pa.flight.FlightUnavailableError("Connect failed")
    )
    mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(RuntimeError):
        client._wait_on_healthcheck(n_tries=2, sleep_n=0)


# ---------------------------------------------------------------------------
# sleep is called with the right value per attempt
# ---------------------------------------------------------------------------


def test_healthcheck_sleeps_between_attempts(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(
        side_effect=[ValueError("nope"), ValueError("nope"), iter([])]
    )
    mock_sleep = mocker.patch("xorq.flight.client.time.sleep")

    client._wait_on_healthcheck(n_tries=5, sleep_n=3)

    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(3)


# ---------------------------------------------------------------------------
# RuntimeError after exhausting n_tries
# ---------------------------------------------------------------------------


def test_healthcheck_raises_runtime_error_after_n_tries(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(side_effect=ValueError("down"))
    mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(RuntimeError, match="failed to connect after"):
        client._wait_on_healthcheck(n_tries=3, sleep_n=0)


def test_healthcheck_runtime_error_attempt_count_in_message(mocker):
    client = _make_client()
    client.do_action = mocker.MagicMock(side_effect=ValueError("down"))
    mocker.patch("xorq.flight.client.time.sleep")

    with pytest.raises(RuntimeError, match="3 attempts"):
        client._wait_on_healthcheck(n_tries=3, sleep_n=0)
