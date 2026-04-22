import json
import logging
import logging.handlers
import pathlib
from unittest.mock import MagicMock

import pytest
import structlog
from opentelemetry.trace import StatusCode
from structlog.testing import LogCapture

from xorq.common.utils.logging_utils import (
    RunLogger,
    _NullRunLogger,
    get_log_path,
    log_initial_state,
)


@pytest.fixture(name="log_output")
def fixture_log_output():
    return LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    structlog.configure(processors=[log_output])


def _has_event(events, event_name):
    return any(entry.get("event", "") == event_name for entry in events)


def test_logging_with_git(log_output):
    log_initial_state()

    assert _has_event(log_output.entries, "git state")
    assert not _has_event(log_output.entries, "xorq version")


def test_logging_without_git(log_output, tmp_path):
    log_initial_state(cwd=tmp_path)

    assert not _has_event(log_output.entries, "git state")
    assert _has_event(log_output.entries, "xorq version")


def test_temp_log_path():
    bad_log_path = "/nonexistantandunwritablepath/"
    log_path = pathlib.Path(get_log_path(bad_log_path))
    assert log_path.exists()
    with pytest.raises(ValueError):
        log_path.relative_to(bad_log_path)


def test_log_span_event_writes_to_both(tmp_path):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    rl = RunLogger(run_dir=run_dir)
    span = MagicMock()

    rl.log_span_event(span, "my.event", {"key": "val"})

    span.add_event.assert_called_once_with("my.event", {"key": "val"})
    events = [
        json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event"] == "my.event"
    assert events[0]["key"] == "val"


def test_log_span_event_none_span(tmp_path):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    rl = RunLogger(run_dir=run_dir)

    rl.log_span_event(None, "my.event", {"k": 1})

    events = [
        json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event"] == "my.event"


def test_log_event_none_fields(tmp_path):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    rl = RunLogger(run_dir=run_dir)

    rl.log_event("bare.event")

    events = [
        json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()
    ]
    assert events[0]["event"] == "bare.event"


def test_from_expr_hash_sets_span_ok(tmp_path):
    span = MagicMock()
    span.get_span_context.return_value = None

    with RunLogger.from_expr_hash("hash1", runs_dir=tmp_path, span=span) as rl:
        rl.log_event("inside")

    span.set_status.assert_called_once_with(StatusCode.OK)
    span.record_exception.assert_not_called()


def test_from_expr_hash_sets_span_error(tmp_path):
    span = MagicMock()
    span.get_span_context.return_value = None

    with pytest.raises(RuntimeError, match="boom"):
        with RunLogger.from_expr_hash("hash2", runs_dir=tmp_path, span=span):
            raise RuntimeError("boom")

    span.set_status.assert_called_once_with(StatusCode.ERROR, "boom")
    span.record_exception.assert_called_once()


def test_from_expr_hash_no_span(tmp_path):
    with RunLogger.from_expr_hash("hash3", runs_dir=tmp_path) as rl:
        rl.log_event("no_span")

    meta = json.loads((rl.run_dir / "meta.json").read_text())
    assert meta["status"] == "ok"


def test_from_expr_hash_finalizes_with_span_context(tmp_path):
    span = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.trace_id = 0x1234567890ABCDEF1234567890ABCDEF
    mock_ctx.is_valid = True
    span.get_span_context.return_value = mock_ctx

    with RunLogger.from_expr_hash("hash4", runs_dir=tmp_path, span=span) as rl:
        pass

    meta = json.loads((rl.run_dir / "meta.json").read_text())
    assert meta["status"] == "ok"
    assert "otel_trace_id" in meta


def test_rotating_file_handler_rotates(tmp_path):
    log_file = tmp_path / "xorq.log"
    max_bytes = 1024

    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=3
    )
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[structlog.processors.JSONRenderer()],
        )
    )
    test_logger = logging.getLogger("test_rotation")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )

    logger = structlog.get_logger("test_rotation")
    for i in range(200):
        logger.info("padding", i=i, data="x" * 50)

    assert log_file.exists()
    assert (tmp_path / "xorq.log.1").exists()

    test_logger.removeHandler(handler)
    handler.close()


def test_null_run_logger_is_noop():
    nrl = _NullRunLogger()
    nrl.log_event("x", {"a": 1})
    nrl.log_span_event(MagicMock(), "x", {"a": 1})
    nrl.finalize(status="ok", span_context=None, error=None)
