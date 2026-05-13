from __future__ import annotations

import pathlib
import sys
from typing import Any, Optional

from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
    parse_bool_env,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.config import Config
from xorq.vendor.ibis.config import Options as IbisOptions


env_config = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.xorq.template")
).from_env()


class Cache(Config):
    """xorq cache configuration options

    Attributes
    ----------

    default_relative_path : str

    """

    default_relative_path: str | pathlib.Path = pathlib.Path(
        env_config.XORQ_DEFAULT_RELATIVE_PATH
    )
    key_prefix: str = env_config.XORQ_CACHE_KEY_PREFIX


class Interactive(Config):
    """Options controlling the interactive repr."""

    @property
    def max_rows(self) -> int:
        return ibis.options.repr.interactive.max_rows

    @max_rows.setter
    def max_rows(self, value: int):
        ibis.options.repr.interactive.max_rows = value

    @property
    def max_columns(self) -> Optional[int]:
        return ibis.options.repr.interactive.max_columns

    @max_columns.setter
    def max_columns(self, value: Optional[int]):
        ibis.options.repr.interactive.max_columns = value

    @property
    def max_length(self) -> int:
        return ibis.options.repr.interactive.max_length

    @max_length.setter
    def max_length(self, value: int):
        ibis.options.repr.interactive.max_length = value

    @property
    def max_string(self) -> int:
        return ibis.options.repr.interactive.max_string

    @max_string.setter
    def max_string(self, value: int):
        ibis.options.repr.interactive.max_string = value

    @property
    def max_depth(self) -> int:
        return ibis.options.repr.interactive.max_depth

    @max_depth.setter
    def max_depth(self, value: int):
        ibis.options.repr.interactive.max_depth = value

    @property
    def show_types(self) -> bool:
        return ibis.options.repr.interactive.show_types

    @show_types.setter
    def show_types(self, value: bool):
        ibis.options.repr.interactive.show_types = value


class Repr(Config):
    """Expression printing options.

    Attributes
    ----------
    interactive : Interactive
        Options controlling the interactive repr.
    """

    interactive: Interactive = Interactive()


class SQL(Config):
    """SQL-related options.

    Attributes
    ----------
    dialect : str
        Dialect to use for printing SQL when the backend cannot be determined.

    """

    dialect: str = "datafusion"


class Pins(Config):
    """SQL-related options.

    Attributes
    ----------
    dialect : str
        Dialect to use for printing SQL when the backend cannot be determined.

    """

    protocol: str = "gcs"
    path: str = "letsql-pins"
    storage_options: dict[str, Any] = {
        "cache_timeout": 0,
        "token": "anon",
    }

    def get_board(self, **kwargs):
        import pins  # noqa: PLC0415

        _kwargs = {
            **{
                "protocol": self.protocol,
                "path": self.path,
                "storage_options": self.storage_options,
            },
            **kwargs,
        }
        return pins.board(**_kwargs)

    def get_path(self, name, board=None, **kwargs):
        import tempfile  # noqa: PLC0415

        from filelock import FileLock  # noqa: PLC0415

        board = board or self.get_board()

        # Cross-process lock keyed by pin name to prevent concurrent
        # downloads from corrupting cached files under pytest-xdist.
        lock_path = pathlib.Path(tempfile.gettempdir()) / f"xorq-pin-{name}.lock"
        lock = FileLock(lock_path, timeout=120)

        with lock:
            (path,) = board.pin_download(name, **kwargs)

        return path


class TUI(Config):
    """xorq catalog TUI layout options.

    Attributes
    ----------
    left_ratio : int
        Width fraction of the left column (catalog tree side).
    right_ratio : int
        Width fraction of the right column (SQL / Info / Schema side).
    revisions_open : bool
        Whether the Revisions panel is shown at startup.
    git_log_open : bool
        Whether the Git log panel is shown at startup.
    """

    left_ratio: int = max(int(env_config.XORQ_TUI_LEFT_RATIO or 2), 1)
    right_ratio: int = max(int(env_config.XORQ_TUI_RIGHT_RATIO or 3), 1)
    revisions_open: bool = bool(env_config.XORQ_TUI_REVISIONS_OPEN) and parse_bool_env(
        env_config.XORQ_TUI_REVISIONS_OPEN
    )
    git_log_open: bool = bool(env_config.XORQ_TUI_GIT_LOG_OPEN) and parse_bool_env(
        env_config.XORQ_TUI_GIT_LOG_OPEN
    )


def _default_use_hardlink(platform=None, env_value=None):
    """Default for ``options.uv.use_hardlink``: env override wins, else darwin-only."""
    if platform is None:
        platform = sys.platform
    if env_value is None:
        env_value = env_config.XORQ_UV_USE_HARDLINK
    if env_value:
        # .capitalize() so shell-style "true"/"TRUE" parse like Python "True".
        return bool(ast.literal_eval(env_value.capitalize()))
    return platform == "darwin"


class UV(Config):
    """uv subprocess options.

    Attributes
    ----------
    use_hardlink : bool
        Pass ``--link-mode hardlink`` to packager ``uv`` invocations.
        Defaults to True on macOS (avoids syspolicyd rescan; see #1942).
    """

    use_hardlink: bool = _default_use_hardlink()


class Options(IbisOptions):
    """xorq configuration options

    Attributes
    ----------
    cache : Cache
        Options controlling caching.
    default_backend : Optional[xorq.vendor.ibis.backends.BaseBackend]
        The default backend to use for execution. Defaults to a lazily
        initialised xorq_datafusion backend; may be set to any BaseBackend
        instance (e.g. via ``xorq.set_backend``).
    repr : Repr
        Options controlling expression printing.
    tui : TUI
        Options controlling the catalog TUI layout.
    uv : UV
        Options controlling how xorq invokes uv subprocesses.
    """

    cache: Cache = Cache()
    repr: Repr = Repr()
    sql: SQL = SQL()
    pins: Pins = Pins()
    tui: TUI = TUI()
    uv: UV = UV()
    default_backend: Optional[BaseBackend] = None
    debug: bool = bool(env_config.XORQ_DEBUG) and parse_bool_env(env_config.XORQ_DEBUG)

    @property
    def interactive(self) -> bool:
        """Show the first few rows of computing an expression when in a repl."""
        return ibis.options.interactive

    @interactive.setter
    def interactive(self, value: bool):
        ibis.options.interactive = value


options = Options()


def default_backend():
    """Return the lazily initialised default backend (xorq_datafusion)."""
    if (backend := options.default_backend) is not None:
        return backend

    from xorq.backends.xorq_datafusion import connect  # noqa: PLC0415

    con = connect()
    # Pin idx to a stable sentinel so the default backend's profile name is
    # identical across processes and test orderings. Otherwise it varies with
    # whatever the global itertools.count() factory has consumed, and leaks
    # into profiles.yaml, expr.yaml profile refs, and cache keys derived from
    # the default backend.
    con._profile = con._profile.clone(idx=-1)
    options.default_backend = con
    return con
