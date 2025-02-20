import pathlib

import pytest
import structlog
from structlog.testing import LogCapture

from xorq.common.utils.logging_utils import (
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
