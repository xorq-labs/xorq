import threading


class LazyBackend:
    """Wraps an unconnected backend instance and defers ``do_connect()`` until
    the first attribute access.

    This lets you build expression trees and store backend references without
    opening a connection until query execution time.  ``do_connect`` is called
    at most once, and the call is thread-safe.

    Parameters
    ----------
    backend : BaseBackend
        An *unconnected* backend instance (e.g. ``duckdb.Backend()`` — not
        the result of ``duckdb.connect(...)``).
    *args, **kwargs
        Forwarded verbatim to ``backend.do_connect()`` on first attribute
        access.

    Examples
    --------
    >>> import xorq.backends.duckdb as duckdb
    >>> from xorq.backends._lazy import LazyBackend
    >>> lazy_con = LazyBackend(duckdb.Backend(), database=":memory:")
    >>> lazy_con.is_connected
    False
    >>> _ = lazy_con.name          # first access triggers do_connect
    >>> lazy_con.is_connected
    True
    """

    def __init__(self, backend, *args, **kwargs):
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_connected", False)
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def __class__(self):
        return type(object.__getattribute__(self, "_backend"))

    def __getattr__(self, name):
        self._ensure_connected()
        backend = object.__getattribute__(self, "_backend")
        return getattr(backend, name)

    def __setattr__(self, name, value):
        backend = object.__getattribute__(self, "_backend")
        setattr(backend, name, value)

    def __repr__(self):
        backend = object.__getattribute__(self, "_backend")
        connected = object.__getattribute__(self, "_connected")
        status = "connected" if connected else "not connected"
        return f"LazyBackend({type(backend).__name__}, {status})"

    @property
    def is_connected(self) -> bool:
        return object.__getattribute__(self, "_connected")

    def _ensure_connected(self) -> None:
        if not object.__getattribute__(self, "_connected"):
            lock = object.__getattribute__(self, "_lock")
            with lock:
                # Double-checked locking: re-test inside the lock.
                if not object.__getattribute__(self, "_connected"):
                    backend = object.__getattribute__(self, "_backend")
                    args = object.__getattribute__(self, "_args")
                    kwargs = object.__getattribute__(self, "_kwargs")
                    backend.do_connect(*args, **kwargs)
                    object.__setattr__(self, "_connected", True)
