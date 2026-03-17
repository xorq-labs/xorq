import threading


class LazyBackend:
    """Wraps a ``connect`` callable and defers the call until the first
    attribute access.

    This lets you build expression trees and store backend references without
    opening a connection until query execution time.  ``connect`` is called
    at most once, and the call is thread-safe.

    Parameters
    ----------
    connect : callable
        The bound ``connect`` method on an unconnected backend instance
        (e.g. ``duckdb.Backend().connect``).  Calling it returns a fully
        initialised, connected backend.
    *args, **kwargs
        Forwarded verbatim to ``connect()`` on first attribute access.

    Examples
    --------
    >>> import xorq.backends.duckdb as duckdb
    >>> from xorq.backends._lazy import LazyBackend
    >>> lazy_con = LazyBackend(duckdb.Backend().connect, database=":memory:")
    >>> lazy_con.is_connected
    False
    >>> _ = lazy_con.name          # first access triggers connect
    >>> lazy_con.is_connected
    True
    """

    def __init__(self, connect, *args, **kwargs):
        object.__setattr__(self, "_connect", connect)
        if hasattr(connect, "__self__"):
            # plain bound method: raw.connect
            backend_cls = type(connect.__self__)
        else:
            # load_backend closure: connect.__wrapped__ = backend.do_connect
            backend_cls = type(connect.__wrapped__.__self__)
        object.__setattr__(self, "_backend_cls", backend_cls)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_backend", None)
        object.__setattr__(self, "_connected", False)
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def __class__(self):
        backend = object.__getattribute__(self, "_backend")
        if backend is not None:
            return type(backend)
        return object.__getattribute__(self, "_backend_cls")

    def __getattr__(self, name):
        self._ensure_connected()
        backend = object.__getattribute__(self, "_backend")
        return getattr(backend, name)

    def __setattr__(self, name, value):
        backend = object.__getattribute__(self, "_backend")
        if backend is not None:
            setattr(backend, name, value)

    def __repr__(self):
        backend_cls = object.__getattribute__(self, "_backend_cls")
        connected = object.__getattribute__(self, "_connected")
        status = "connected" if connected else "not connected"
        return f"LazyBackend({backend_cls.__name__}, {status})"

    @property
    def is_connected(self) -> bool:
        return object.__getattribute__(self, "_connected")

    def _ensure_connected(self) -> None:
        if not object.__getattribute__(self, "_connected"):
            lock = object.__getattribute__(self, "_lock")
            with lock:
                # Double-checked locking: re-test inside the lock.
                if not object.__getattribute__(self, "_connected"):
                    connect = object.__getattribute__(self, "_connect")
                    args = object.__getattribute__(self, "_args")
                    kwargs = object.__getattribute__(self, "_kwargs")
                    backend = connect(*args, **kwargs)
                    object.__setattr__(self, "_backend", backend)
                    object.__setattr__(self, "_connected", True)
