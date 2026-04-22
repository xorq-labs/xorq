import datetime
import hashlib
import json
import logging.handlers
import pathlib
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from opentelemetry.trace import SpanContext, StatusCode


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

import structlog
from attr import field, frozen
from attr.validators import instance_of

from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


default_log_path = pathlib.Path("~/.config/xorq/xorq.log").expanduser()

_log_env_config = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.xorq.template"),
    prefix="XORQ_",
).from_env()


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
            import xorq  # noqa: PLC0415

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
_log_level_str = _log_env_config.log_level.upper()

if _log_level_str != "OFF":
    log_level = getattr(logging, _log_level_str)
    _xorq_logger = logging.getLogger("xorq")
    _xorq_logger.setLevel(log_level)
    _rfh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 2**20, backupCount=3
    )
    _rfh.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    _xorq_logger.addHandler(_rfh)
else:
    log_level = logging.CRITICAL

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(log_level),
)
get_logger = structlog.get_logger

if _log_level_str != "OFF":
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
#         <run_id>/            # e.g. 550e8400-e29b-41d4-a716-446655440000
#           run.jsonl          # append-only event log (one JSON object per line)
#           meta.json          # summary written on completion
#
# The run ID is a UUID4, unique across all runs.


class RunLogFile(StrEnum):
    LOG = "run.jsonl"
    META = "meta.json"


def get_xorq_runs_dir() -> Path:
    # NOTE: modifying env var XORQ_RUNS_LOGS_DIR won't have any impact after first import
    from xorq.config import env_config  # noqa: PLC0415

    if path := env_config.XORQ_RUNS_LOGS_DIR:
        return Path(path).expanduser()
    return Path("~/.local/share/xorq/runs").expanduser()


@frozen
class RunLogger:
    """Writes structured events to run.jsonl and a summary to meta.json."""

    run_dir: Path = field(validator=instance_of(Path))
    params_tuple = field(validator=instance_of(tuple), default=())
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
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def _log_path(self) -> Path:
        return self.run_dir / RunLogFile.LOG

    @property
    def _meta_path(self) -> Path:
        return self.run_dir / RunLogFile.META

    @property
    def _finalized(self) -> bool:
        return self._meta_path.exists()

    def log_event(self, event: str, fields: dict = None):
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            **(fields or {}),
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_span_event(self, span, event: str, fields: dict = None):
        """Log to both the run log and an OTel span."""
        self.log_event(event, fields)
        if span is not None:
            span.add_event(event, fields or {})

    def finalize(
        self,
        status: str,
        span_context: SpanContext = None,
        error: str = None,
    ):
        """Write meta.json and close the log file. Idempotent."""
        if self._finalized:
            return

        meta = {
            "run_id": self.run_id,
            "started_at": self._started_at,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": status,
            **dict(self.params_tuple),
            **(
                {"otel_trace_id": otel_trace_id}
                if (otel_trace_id := RunLogger._get_otel_trace_id(span_context))
                is not None
                else {}
            ),
            **({"error": error} if error is not None else {}),
        }
        try:
            self._fh.close()
            self._meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        except Exception:
            pass

    @staticmethod
    def _make_run_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _compute_file_metrics(output_format, output_path) -> dict:
        metrics = {}
        if output_path and output_path != "-":
            output_file = Path(output_path)
            if output_file.exists():
                metrics["bytes"] = output_file.stat().st_size
                if str(output_format) == "parquet":
                    try:
                        import pyarrow.parquet as pq  # noqa: PLC0415

                        metrics["rows"] = pq.read_metadata(output_file).num_rows
                    except ImportError:
                        pass
        return metrics

    @staticmethod
    def _get_otel_trace_id(span_ctx: SpanContext) -> str | None:
        otel_trace_id = (
            format(span_ctx.trace_id, "032x")
            if span_ctx and span_ctx.is_valid
            else None
        )
        return otel_trace_id

    @classmethod
    @contextmanager
    def from_expr_hash(
        cls, expr_hash: str, *, params_tuple: tuple = None, runs_dir=None, span=None
    ):
        """Context manager that creates a :class:`RunLogger` and finalizes it on exit.

        If the run store directory cannot be created (e.g. permission error), a
        no-op :class:`_NullRunLogger` is yielded so the actual run is not affected.

        When *span* is provided, sets its status on exit and records exceptions,
        so callers don't need a separate try/except for span bookkeeping.
        """
        try:
            runs_dir_path = (
                Path(runs_dir) if runs_dir is not None else get_xorq_runs_dir()
            )
            run_id = cls._make_run_id()
            run_dir = runs_dir_path / expr_hash / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            rl = cls(
                run_dir=run_dir,
                params_tuple=params_tuple or (("expr_hash", expr_hash),),
            )
        except IOError:
            rl = _NullRunLogger()

        error_msg = None
        try:
            yield rl
        except Exception as exc:
            error_msg = str(exc)
            if span is not None:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
            raise
        else:
            if span is not None:
                span.set_status(StatusCode.OK)
        finally:
            span_context = span.get_span_context() if span is not None else None
            rl.finalize(
                status="error" if error_msg else "ok",
                error=error_msg,
                span_context=span_context,
            )


class _NullRunLogger:
    """No-op RunLogger used when the run store cannot be initialized."""

    run_id = None
    run_dir = None

    def log_event(self, event: str, fields: dict = None):
        pass

    def log_span_event(self, span, event: str, fields: dict = None):
        pass

    def finalize(self, **kwargs):
        pass


@frozen
class Run:
    """A single recorded run, readable from disk."""

    run_dir: Path = field(validator=instance_of(Path))

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def _meta_path(self) -> Path:
        return self.run_dir / RunLogFile.META

    @property
    def _log_path(self) -> Path:
        return self.run_dir / RunLogFile.LOG

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
    """Run store for a given expression directory."""

    expr_dir: Path = field(validator=instance_of(Path))

    @property
    def runs(self) -> tuple[Run, ...]:
        """Return Run objects most recent first."""
        return tuple(Run(run_dir=self.expr_dir / run_id) for run_id in self.list())

    def list(self) -> tuple[str, ...]:
        """Return run IDs most recent first (cheap: no Run objects created)."""
        if not self.expr_dir.exists():
            return ()
        return tuple(
            p.name
            for p in sorted(
                (p for p in self.expr_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )
