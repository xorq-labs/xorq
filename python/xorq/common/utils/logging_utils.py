from __future__ import annotations

import datetime
import functools
import hashlib
import importlib.metadata
import json
import logging.handlers
import os
import pathlib
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from attr import field, frozen


if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext

from attr.validators import instance_of

from xorq.common.enums import RunLogFile
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


default_log_path = pathlib.Path("~/.config/xorq/xorq.log").expanduser()

_log_env_config = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.xorq.template"),
    prefix="XORQ_",
).from_env()


# a hung git (network filesystem, stuck fsmonitor) must not hang the import
# or the build that records its state
GIT_TIMEOUT_SECONDS = 30
GIT_FAILURE_EVENT = "failed to read git state"


def _git(args, cwd=None):
    # bytes: a diff can carry any encoding, so each caller decodes
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, timeout=GIT_TIMEOUT_SECONDS
    )


def _has_dot_git_ancestor(cwd=None):
    # resolved, as git resolves it: a symlink into a repo's subdirectory is
    # inside that repo, whatever its own lexical parents are
    try:
        start = Path(cwd or ".").resolve()
    except RuntimeError:
        # a symlink loop, before Python 3.13; it leads to no directory
        return False
    return any(path.joinpath(".git").exists() for path in (start, *start.parents))


def _git_stderr(error_or_completed):
    # git's own explanation lives in stderr, which an exception's message
    # leaves out; a timeout or an OSError carries none
    stderr = getattr(error_or_completed, "stderr", None) or b""
    return stderr.decode("utf-8", errors="replace").strip()


def _git_is_present(cwd=None):
    """True when `cwd` is inside a git work tree, commits or not.

    Git only runs below a `.git` entry, so `import xorq` outside any repo
    forks nothing. A missing `cwd`, a bare repo and a directory outside any
    repo answer False; so do a missing git binary, a repo git refuses to
    read (ownership, corruption) and a git that times out, which also log
    why.
    """
    logger = structlog.get_logger(__name__)
    try:
        # an unreadable ancestor raises from the `.git` lookup too
        if not _has_dot_git_ancestor(cwd=cwd):
            return False
        inside = _git(["rev-parse", "--is-inside-work-tree"], cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.exception(GIT_FAILURE_EVENT, git_stderr=_git_stderr(e))
        return False
    if inside.returncode != 0:
        logger.warning(GIT_FAILURE_EVENT, git_stderr=_git_stderr(inside))
    return inside.stdout.strip() == b"true"


def _git_head(cwd=None):
    """The commit HEAD points at, or None on an unborn branch (no commit yet)."""
    head = _git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=cwd)
    return head.stdout.decode().strip() if head.returncode == 0 else None


# the same patch bytes whatever the user's config and wherever in the work
# tree git runs: each flag pins a diff.* setting (external driver, textconv,
# color, relative paths, prefixes, context, hunk joining, algorithm, rename
# detection, file order, submodule format, object id length), and the -c
# pairs pin how non-ASCII paths and blank context lines are written
PLAIN_DIFF = (
    "-c",
    "core.quotePath=true",
    "-c",
    "diff.suppressBlankEmpty=false",
    "diff",
    "--full-index",
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--no-relative",
    "--src-prefix=a/",
    "--dst-prefix=b/",
    "-U3",
    "--inter-hunk-context=0",
    "--diff-algorithm=myers",
    "--indent-heuristic",
    "--no-renames",
    f"-O{os.devnull}",
    "--submodule=short",
)


def _git_output(args, cwd=None):
    # decoded as UTF-8 with replacement: a diff must not abort the build that
    # records it, while valid UTF-8 and CRLF content stay exactly as git
    # printed them (text=True would decode with the locale and fold newlines)
    completed = _git(args, cwd=cwd)
    completed.check_returncode()
    return completed.stdout.decode("utf-8", errors="replace").strip()


def get_git_state(hash_diffs, cwd=None):
    # both diffs work against the empty tree on an unborn branch, so a fresh
    # `git init; git add` project records what is staged
    git_state = {
        "commit": _git_head(cwd=cwd),
        "diff": _git_output(PLAIN_DIFF, cwd=cwd),
        "diff_cached": _git_output([*PLAIN_DIFF, "--cached"], cwd=cwd),
    }
    if hash_diffs:
        for key in ("diff", "diff_cached"):
            git_state[f"{key}_hash"] = hashlib.md5(
                git_state.pop(key).encode()
            ).hexdigest()
    return git_state


def git_state_or_none(hash_diffs=False, cwd=None):
    """The git state of `cwd`, or None outside a work tree or when git fails.

    Recording the git state is best effort: a git error is logged and never
    aborts the build or the import that asked for it.
    """
    if not _git_is_present(cwd=cwd):
        return None
    try:
        return get_git_state(hash_diffs=hash_diffs, cwd=cwd)
    except (subprocess.SubprocessError, OSError) as e:
        structlog.get_logger(__name__).exception(
            GIT_FAILURE_EVENT, git_stderr=_git_stderr(e)
        )
        return None


@functools.cache
def _get_xorq_version() -> str:
    try:
        return importlib.metadata.version("xorq")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def log_initial_state(hash_diffs=False, cwd=None):
    logger = structlog.get_logger(__name__)
    logger.info("initial log level", log_level=log_level)
    if (git_state := git_state_or_none(hash_diffs=hash_diffs, cwd=cwd)) is not None:
        logger.info("git state", **git_state)
    else:
        logger.info("xorq version", version=_get_xorq_version())


def get_log_path(log_path=default_log_path):
    try:
        log_path.parent.mkdir(exist_ok=True, parents=True)
    except Exception:
        _, log_path = tempfile.mkstemp(suffix=".log", prefix="xorq-")
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
_configured_log_path = pathlib.Path(_log_env_config.log_path).expanduser()
log_path = get_log_path(log_path=_configured_log_path)
_log_level_str = _log_env_config.log_level.upper()

if _log_level_str != "OFF":
    log_level = getattr(logging, _log_level_str)
    _xorq_logger = logging.getLogger("xorq")
    _xorq_logger.setLevel(log_level)
    _MAX_LOG_BYTES = 50 * 2**20
    try:
        _rfh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=_MAX_LOG_BYTES, backupCount=3
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
    except Exception as e:
        import warnings  # noqa: PLC0415

        warnings.warn(f"xorq: file logging disabled: {e}", stacklevel=2)
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
            span.add_event(
                event,
                {k: v for k, v in (fields or {}).items() if v is not None},
            )

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
            "xorq_version": _get_xorq_version(),
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
            if output_file.is_file():
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
                from xorq.common.utils.otel_utils import set_span_error  # noqa: PLC0415

                set_span_error(span, exc)
            raise
        else:
            if span is not None:
                from xorq.common.utils.otel_utils import set_span_ok  # noqa: PLC0415

                set_span_ok(span)
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
