import json
import logging
import logging.handlers
import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import structlog
from opentelemetry.trace import StatusCode
from structlog.testing import LogCapture

import xorq.api as xo
import xorq.common.utils.logging_utils as logging_utils
from xorq.common.utils.logging_utils import (
    RunLogger,
    _NullRunLogger,
    get_log_path,
    log_initial_state,
)
from xorq.ibis_yaml.compiler import build_expr
from xorq.ibis_yaml.enums import DumpFiles


@pytest.fixture(name="log_output")
def fixture_log_output():
    return LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    # restore xorq's configuration afterwards: structlog is global, and a
    # LogCapture left behind swallows the warnings the next test file on the
    # same worker asserts through caplog
    saved = structlog.get_config()
    structlog.configure(processors=[log_output])
    yield
    structlog.configure(**saved)


# git reads these before it looks at the cwd: exported by a git hook or a
# developer's shell, they would point the tmp repos, and the logger under
# test, at another repository
GIT_REPO_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
)


@pytest.fixture(autouse=True)
def fixture_isolated_git(monkeypatch):
    for name in GIT_REPO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # no global or system config: no hooksPath, templateDir or diff driver
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


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


def test_import_outside_repo_logs_version(tmp_path, monkeypatch):
    # the module logs its initial state while it is still being imported
    monkeypatch.setenv("XORQ_LOG_LEVEL", "INFO")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import xorq.common.utils.logging_utils",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _git(cwd, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=xorq",
            "-c",
            "user.email=xorq@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _build_git_state(builds_dir):
    build_path = build_expr(xo.memtable({"x": [1]}), builds_dir=builds_dir)
    metadata = json.loads(
        Path(build_path).joinpath(DumpFiles.build_metadata).read_text()
    )
    return metadata["git_state"]


@pytest.fixture(name="empty_repo")
def fixture_empty_repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_build_git_state_without_commits(empty_repo, log_output):
    # GH #1561: a freshly `git init`ed project has a .git directory but no HEAD
    assert _build_git_state(empty_repo / "builds") == {
        "commit": None,
        "diff": "",
        "diff_cached": "",
    }

    log_initial_state(cwd=empty_repo)
    assert _has_event(log_output.entries, "git state")
    assert not _has_event(log_output.entries, "failed to read git state")


def test_build_git_state_staged_without_commits(empty_repo):
    # the tutorial flow: git init, git add, build; what is staged is recorded
    (empty_repo / "pipeline.py").write_text("expr = None\n")
    _git(empty_repo, "add", "pipeline.py")

    git_state = _build_git_state(empty_repo / "builds")
    assert git_state["commit"] is None
    assert "pipeline.py" in git_state["diff_cached"]
    assert git_state["diff"] == ""


def test_git_state_follows_cwd_not_process_cwd(tmp_path, log_output, monkeypatch):
    # the logger is pointed at a repo from a directory that is not one
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "first")
    monkeypatch.chdir(tmp_path)

    log_initial_state(cwd=repo)
    (event,) = (e for e in log_output.entries if e["event"] == "git state")
    assert event["commit"] == _git(repo, "rev-parse", "HEAD")
    assert not _has_event(log_output.entries, "failed to read git state")


def test_build_git_state_staged_non_utf8_file(empty_repo):
    (empty_repo / "latin1.txt").write_bytes("se\xf1or\n".encode("latin-1"))
    _git(empty_repo, "add", "latin1.txt")

    git_state = _build_git_state(empty_repo / "builds")
    assert "latin1.txt" in git_state["diff_cached"]


def test_build_git_state_keeps_crlf_and_utf8(empty_repo):
    (empty_repo / "crlf.txt").write_bytes("se\u00f1or\r\nsecond\r\n".encode("utf-8"))
    _git(empty_repo, "-c", "core.autocrlf=false", "add", "crlf.txt")

    diff_cached = _build_git_state(empty_repo / "builds")["diff_cached"]
    assert "+se\u00f1or\r\n+second" in diff_cached


def test_build_git_state_with_commit(empty_repo):
    _git(empty_repo, "commit", "-q", "--allow-empty", "-m", "first")

    git_state = _build_git_state(empty_repo / "builds")
    assert git_state["commit"] == _git(empty_repo, "rev-parse", "HEAD")


def test_build_survives_git_failure(empty_repo, log_output):
    # inside a work tree, but `git diff` cannot read the index
    _git(empty_repo, "commit", "-q", "--allow-empty", "-m", "first")
    (empty_repo / ".git" / "index").write_bytes(b"garbage")

    assert _build_git_state(empty_repo / "builds") is None

    log_initial_state(cwd=empty_repo)
    assert _has_event(log_output.entries, "failed to read git state")
    assert _has_event(log_output.entries, "xorq version")


def _fake_git_on_path(tmp_path, monkeypatch, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(f"#!/bin/sh\n{body}\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)
    return fake_git


def test_no_git_runs_outside_any_repo(tmp_path, monkeypatch, log_output):
    marker = tmp_path / "git-ran"
    _fake_git_on_path(tmp_path, monkeypatch, f"touch {marker}")
    outside = tmp_path / "outside"
    outside.mkdir()

    log_initial_state(cwd=outside)
    assert not marker.exists()
    assert _has_event(log_output.entries, "xorq version")


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, which root bypasses",
)
def test_unreadable_directory_logs_version(tmp_path, log_output):
    locked = tmp_path / "locked"
    inner = locked / "inner"
    inner.mkdir(parents=True)
    locked.chmod(0)
    try:
        log_initial_state(cwd=inner)
    finally:
        locked.chmod(0o755)
    assert _has_event(log_output.entries, "xorq version")


def test_hung_git_times_out(tmp_path, monkeypatch, log_output):
    repo = tmp_path / "repo"
    repo.joinpath(".git").mkdir(parents=True)
    _fake_git_on_path(tmp_path, monkeypatch, "exec sleep 30")
    monkeypatch.setattr(logging_utils, "GIT_TIMEOUT_SECONDS", 0.2)

    started = time.monotonic()
    log_initial_state(cwd=repo)
    assert time.monotonic() - started < 10
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


@pytest.mark.library
def test_log_span_event_drops_none_attributes_from_span(tmp_path):
    # Regression test for #1940: span.add_event must drop None attrs
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    rl = RunLogger(run_dir=run_dir)
    span = MagicMock()

    rl.log_span_event(span, "my.event", {"good": "val", "limit": None, "ttl": None})

    span.add_event.assert_called_once_with("my.event", {"good": "val"})

    events = [
        json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["good"] == "val"
    assert events[0]["limit"] is None
    assert events[0]["ttl"] is None


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
