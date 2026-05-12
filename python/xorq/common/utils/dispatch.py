"""Single-dispatch registry with lazy module-load support.

Direct port of dask.utils.Dispatch so xorq can drop its dask runtime dependency.
"""

from __future__ import annotations


class Dispatch:
    """Simple single dispatch."""

    def __init__(self, name=None):
        self._lookup = {}
        self._lazy = {}
        if name:
            self.__name__ = name

    def register(self, type, func=None):
        """Register dispatch of ``func`` on arguments of type ``type``."""

        def wrapper(func):
            if isinstance(type, tuple):
                for t in type:
                    self.register(t, func)
            else:
                self._lookup[type] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def register_lazy(self, toplevel, func=None):
        """Register a registration function called when ``toplevel`` loads."""

        def wrapper(func):
            self._lazy[toplevel] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def dispatch(self, cls):
        """Return the function implementation for the given ``cls``."""
        lk = self._lookup
        for cls2 in cls.__mro__:
            toplevel, _, _ = cls2.__module__.partition(".")
            try:
                register = self._lazy[toplevel]
            except KeyError:
                pass
            else:
                register()
                self._lazy.pop(toplevel, None)
                return self.dispatch(cls)
            try:
                impl = lk[cls2]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    lk[cls] = impl
                return impl
        raise TypeError(f"No dispatch for {cls}")

    def __call__(self, arg, *args, **kwargs):
        meth = self.dispatch(type(arg))
        return meth(arg, *args, **kwargs)
