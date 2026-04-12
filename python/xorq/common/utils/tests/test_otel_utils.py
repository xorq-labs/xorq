from unittest.mock import MagicMock, patch

from xorq.common.utils.otel_utils import (
    _should_use_otlp_exporter,
    is_localhost_collector_listening,
)


# -- is_localhost_collector_listening ------------------------------------------


@patch("socket.create_connection")
def test_localhost_listening_returns_true_when_connection_succeeds(mock_conn):
    mock_conn.return_value.__enter__ = MagicMock()
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    assert is_localhost_collector_listening("http://localhost:4318/v1/traces")


@patch("socket.create_connection", side_effect=ConnectionRefusedError)
def test_localhost_listening_returns_false_when_connection_refused(mock_conn):
    assert not is_localhost_collector_listening("http://localhost:4318/v1/traces")


@patch("socket.create_connection", side_effect=TimeoutError)
def test_localhost_listening_returns_false_on_timeout(mock_conn):
    assert not is_localhost_collector_listening("http://localhost:4318/v1/traces")


def test_localhost_listening_returns_false_for_remote_host():
    assert not is_localhost_collector_listening("http://remote:4318/v1/traces")


def test_localhost_listening_returns_false_for_missing_port():
    assert not is_localhost_collector_listening("http://localhost/v1/traces")


def test_localhost_listening_returns_false_for_ip_address():
    assert not is_localhost_collector_listening("http://127.0.0.1:4318/v1/traces")


# -- _should_use_otlp_exporter ------------------------------------------------


def test_should_use_otlp_returns_false_for_empty_endpoint():
    assert not _should_use_otlp_exporter("")
    assert not _should_use_otlp_exporter(None)


@patch(
    "xorq.common.utils.otel_utils.is_localhost_collector_listening",
)
def test_should_use_otlp_returns_true_for_remote_endpoint_without_probing(mock_probe):
    assert _should_use_otlp_exporter("http://remote:4318/v1/traces")
    mock_probe.assert_not_called()


@patch(
    "xorq.common.utils.otel_utils.is_localhost_collector_listening",
    return_value=True,
)
def test_should_use_otlp_delegates_to_probe_for_localhost_running(mock_probe):
    assert _should_use_otlp_exporter("http://localhost:4318/v1/traces")
    mock_probe.assert_called_once_with("http://localhost:4318/v1/traces")


@patch(
    "xorq.common.utils.otel_utils.is_localhost_collector_listening",
    return_value=False,
)
def test_should_use_otlp_delegates_to_probe_for_localhost_not_running(mock_probe):
    assert not _should_use_otlp_exporter("http://localhost:4318/v1/traces")
    mock_probe.assert_called_once_with("http://localhost:4318/v1/traces")
