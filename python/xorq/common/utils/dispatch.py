"""String-keyed single dispatch on fully qualified type names.

Dispatches on the first argument's type by walking its MRO and matching
against registered FQN strings. No imports of target types are needed
at registration time.
"""

from __future__ import annotations


def _fqn(typ):
    return f"{typ.__module__}.{typ.__qualname__}"


class FQNDispatch:
    """Declarative single dispatch keyed by fully qualified type name strings.

    Rules are (fqn_string, handler) pairs. Lookup walks the first argument's
    MRO, converting each class to its FQN, and returns the first match.
    """

    def __init__(self, rules, *, default=None):
        self._rules = dict(rules)
        self._default = default
        self._cache = {}

    def __call__(self, arg, *args, **kwargs):
        cls = type(arg)
        handler = self._cache.get(cls)
        if handler is not None:
            return handler(arg, *args, **kwargs)
        for mro_cls in cls.__mro__:
            key = _fqn(mro_cls)
            handler = self._rules.get(key)
            if handler is not None:
                self._cache[cls] = handler
                return handler(arg, *args, **kwargs)
        if self._default is not None:
            return self._default(arg, *args, **kwargs)
        raise TypeError(f"No dispatch for {cls}")

    def register(self, typ_or_fqn, handler):
        """Register ``handler`` for an estimator type at runtime.

        Accepts either a type (whose FQN is computed via ``_fqn``) or a
        pre-computed FQN string.  Clears the MRO-lookup cache so existing
        subclasses re-resolve through the new rule.  Returns ``self`` for
        chaining.

        Preserves the extension point that the legacy dask-derived
        ``Dispatch`` exposed via ``get_predict_return_type.register``.
        """
        key = typ_or_fqn if isinstance(typ_or_fqn, str) else _fqn(typ_or_fqn)
        self._rules[key] = handler
        self._cache.clear()
        return self

    @property
    def registered_fqns(self):
        return tuple(self._rules.keys())
