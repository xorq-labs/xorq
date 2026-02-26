import datetime
import hashlib
import json
import logging.handlers
import os
import pathlib
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import structlog
from attr import field, frozen
from attr.validators import instance_of


default_log_path = pathlib.Path("~/.config/xorq/xorq.log").expanduser()


def _git_is_present(cwd=None):
    if cwd is None:
        cwd = pathlib.Path().absolute()

    return any(p for p in (cwd, *cwd.parents) if p.joinpath(".git").exists())


def get_git_state(hash_diffs):
    (commit, diff, diff_cached) = (
        subprocess.check_output(lst).decode().strip()
        for lst in (
            ["git", "rev-parse", "HEAD"],
            ["git", "diff"],
            ["git", "diff", "--cached"],
        )
    )
    git_state = {
        "commit": commit,
        "diff": diff,
        "diff_cached": diff_cached,
    }
    if hash_diffs:
        for key in ("diff", "diff_cached"):
            git_state[f"{key}_hash"] = hashlib.md5(
                git_state.pop(key).encode()
            ).hexdigest()
    return git_state


def log_initial_state(hash_diffs=False, cwd=None):
    logger = structlog.get_logger(__name__)
    logger.info("initial log level", log_level=log_level)
    try:
        if _git_is_present(cwd=cwd):
            git_state = get_git_state(hash_diffs=hash_diffs)
            logger.info(
                "git state",
                **git_state,
            )
        else:
            import xorq

            logger.info("xorq version", version=xorq.__version__)
    except Exception:
        logger.exception("failed to log git repo info")


def get_log_path(log_path=default_log_path):
    try:
        log_path.parent.mkdir(exist_ok=True, parents=True)
    except Exception:
        (_, log_path) = tempfile.mkstemp(suffix=".log", prefix="xorq-")
    return log_path


def get_print_logger():
    logger = structlog.wrap_logger(
        structlog.PrintLogger(),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )
    return logger


# https://betterstack.com/community/guides/logging/structlog/
log_path = get_log_path(log_path=default_log_path)
log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper())
rfh = logging.handlers.RotatingFileHandler(log_path, maxBytes=50 * 2**20)
structlog.configure(
    logger_factory=structlog.WriteLoggerFactory(rfh._open()),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(log_level),
)
get_logger = structlog.get_logger
log_initial_state()


# ---------------------------------------------------------------------------
# Persistent run logging
# ---------------------------------------------------------------------------
# Each invocation of ``xorq run`` is assigned a unique run ID and its events
# are written to disk so they can be inspected later.
#
# Directory layout::
#
#     ~/.local/share/xorq/runs/
#       <expr_hash>/
#         <run_id>/            # e.g. abc123ef-20260226T143022Z
#           run.jsonl          # append-only event log (one JSON object per line)
#           meta.json          # summary written on completion
#
# The run ID is ``<expr_hash>-<UTC timestamp>``, making runs sortable by time
# while remaining tied to the expression that produced them.


def get_xorq_runs_dir() -> Path:
    # NOTE: modifying env var XORQ_RUNS_LOGS_DIR won't have any impact after first import
    from xorq.config import env_config

    if path := env_config.XORQ_RUNS_LOGS_DIR:
        return Path(path).expanduser()
    return Path("~/.local/share/xorq/runs").expanduser()


def _make_run_id(expr_hash: str) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{expr_hash}-{ts}"


@frozen
class RunLogger:
    """Writes structured events to run.jsonl and a summary to meta.json."""

    run_id: str = field(validator=instance_of(str))
    run_dir: Path = field(validator=instance_of(Path))
    _params: dict = field(validator=instance_of(dict))
    _started_at = field(init=False)
    _fh = field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "_started_at",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        object.__setattr__(self, "_fh", self._log_path.open("a", encoding="utf-8"))

    @property
    def _log_path(self) -> Path:
        return self.run_dir / "run.jsonl"

    @property
    def _meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def _finalized(self) -> bool:
        return self._meta_path.exists()

    def log_event(self, event: str, **fields):
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def finalize(self, status: str, otel_trace_id: str = None, error: str = None):
        """Write meta.json and close the log file. Idempotent."""
        if self._finalized:
            return
        try:
            self._fh.close()
        except Exception:
            pass
        meta = {
            "run_id": self.run_id,
            "started_at": self._started_at,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": status,
            **self._params,
            **({"otel_trace_id": otel_trace_id} if otel_trace_id is not None else {}),
            **({"error": error} if error is not None else {}),
        }
        self._meta_path.write_text(json.dumps(meta, indent=2) + "\n")


class _NullRunLogger:
    """No-op RunLogger used when the run store cannot be initialized."""

    run_id = None
    run_dir = None

    def log_event(self, event: str, **fields):
        pass

    def finalize(
        self, status: str = "ok", otel_trace_id: str = None, error: str = None
    ):
        pass


@contextmanager
def run_logger(expr_hash: str, params: dict, runs_dir=None):
    """Context manager that creates a :class:`RunLogger` and finalizes it on exit.

    If the run store directory cannot be created (e.g. permission error), a
    no-op :class:`_NullRunLogger` is yielded so the actual run is not affected.

    On successful exit the caller is expected to call
    ``rl.finalize(status="ok", otel_trace_id=...)`` explicitly so the OTel
    trace ID can be recorded.  The context manager's ``finally`` block calls
    ``finalize`` only if it has not already been called (idempotent guard on
    ``_finalized``).
    """
    try:
        runs_dir_path = Path(runs_dir) if runs_dir is not None else get_xorq_runs_dir()
        run_id = _make_run_id(expr_hash)
        run_dir = runs_dir_path / expr_hash / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rl = RunLogger(run_id=run_id, run_dir=run_dir, params=params)
    except Exception:
        rl = _NullRunLogger()

    error_msg = None
    try:
        yield rl
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        rl.finalize(
            status="error" if error_msg else "ok",
            error=error_msg,
        )


@frozen
class Run:
    """A single recorded run, readable from disk."""

    run_id: str = field(validator=instance_of(str))
    expr_hash: str = field(validator=instance_of(str))
    runs_dir: Path = field(validator=instance_of(Path))

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.expr_hash / self.run_id

    @property
    def _meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    @property
    def _log_path(self) -> Path:
        return self.run_dir / "run.jsonl"

    def read_meta(self) -> dict | None:
        """Read ``meta.json`` for this run, or ``None`` if not yet written."""
        if not self._meta_path.exists():
            return None
        return json.loads(self._meta_path.read_text())

    def read_events(self) -> tuple[dict, ...]:
        """Read all events from ``run.jsonl`` for this run."""
        if not self._log_path.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self._log_path.read_text().splitlines()
            if line.strip()
        )


@frozen
class Runs:
    """Run store for a given runs directory."""

    runs_dir: Path = field(validator=instance_of(Path))

    def list(self, expr_hash: str) -> tuple[Run, ...]:
        """Return runs for this expression hash, most recent first."""
        expr_dir = self.runs_dir / expr_hash
        if not expr_dir.exists():
            return ()
        return tuple(
            Run(run_id=p.name, expr_hash=expr_hash, runs_dir=self.runs_dir)
            for p in sorted(
                (p for p in expr_dir.iterdir() if p.is_dir()),
                key=lambda p: p.name,
                reverse=True,
            )
        )
