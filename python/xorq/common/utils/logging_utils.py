import hashlib
import logging.handlers
import os
import pathlib
import subprocess
import tempfile

import structlog


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
            import xorq as xo

            logger.info("xorq version", version=xo.__version__)
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
