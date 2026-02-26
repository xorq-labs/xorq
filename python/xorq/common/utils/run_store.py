"""
Persistent run storage for xorq run command.

Each invocation of ``xorq run`` is assigned a unique run ID and its events
are written to disk so they can be inspected later.

Directory layout::

    ~/.local/share/xorq/runs/
      <expr_hash>/
        <run_id>/            # e.g. abc123ef-20260226T143022Z
          run.jsonl          # append-only event log (one JSON object per line)
          meta.json          # summary written on completion

The run ID is ``<expr_hash>-<UTC timestamp>``, making runs sortable by time
while remaining tied to the expression that produced them.
"""

import datetime
import json
from contextlib import contextmanager
from pathlib import Path


def get_xorq_runs_dir() -> Path:
    # NOTE: modifying env var XORQ_RUNS_LOGS_DIR won't have any impact after first import
    from xorq.config import env_config

    if path := env_config.XORQ_RUNS_LOGS_DIR:
        return Path(path).expanduser()
    return Path("~/.local/share/xorq/runs").expanduser()


def _make_run_id(expr_hash: str) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{expr_hash}-{ts}"


class RunLogger:
    """Writes structured events to run.jsonl and a summary to meta.json."""

    def __init__(self, run_id: str, run_dir: Path, params: dict):
        self.run_id = run_id
        self.run_dir = run_dir
        self._log_path = run_dir / "run.jsonl"
        self._meta_path = run_dir / "meta.json"
        self._started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._params = params
        self._fh = self._log_path.open("a", encoding="utf-8")
        self._finalized = False

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
        self._finalized = True
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
    _finalized = False

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


# ---------------------------------------------------------------------------
# Read-side helpers (for inspection / future TUI use)
# ---------------------------------------------------------------------------


def _resolve_runs_dir(runs_dir) -> Path:
    return Path(runs_dir) if runs_dir is not None else get_xorq_runs_dir()


def list_expr_hashes(runs_dir=None) -> tuple[str, ...]:
    """Return sorted tuple of expression hashes that have at least one run."""
    runs_dir_path = _resolve_runs_dir(runs_dir)
    if not runs_dir_path.exists():
        return ()
    return tuple(sorted(p.name for p in runs_dir_path.iterdir() if p.is_dir()))


def list_runs(expr_hash: str, runs_dir=None) -> tuple[str, ...]:
    """Return run IDs for an expression hash, most recent first."""
    runs_dir_path = _resolve_runs_dir(runs_dir)
    expr_dir = runs_dir_path / expr_hash
    if not expr_dir.exists():
        return ()
    return tuple(
        sorted(
            (p.name for p in expr_dir.iterdir() if p.is_dir()),
            reverse=True,
        )
    )


def read_meta(expr_hash: str, run_id: str, runs_dir=None) -> dict | None:
    """Read the ``meta.json`` for a specific run, or ``None`` if not found."""
    runs_dir_path = _resolve_runs_dir(runs_dir)
    meta_path = runs_dir_path / expr_hash / run_id / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def read_events(expr_hash: str, run_id: str, runs_dir=None) -> tuple[dict, ...]:
    """Read all events from ``run.jsonl`` for a specific run."""
    runs_dir_path = _resolve_runs_dir(runs_dir)
    log_path = runs_dir_path / expr_hash / run_id / "run.jsonl"
    if not log_path.exists():
        return ()
    return tuple(
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    )
