import inspect
from functools import (
    partial,
)
from importlib import (
    import_module,
)

from cloudpickle import (
    dumps,
    loads,
)
from toolz.functoolz import (
    _sigs,
    has_varargs,
    instanceproperty,
    is_partial_args,
    is_valid_args,
    no_default,
)


class curry(object):
    """Curry a callable function

    Enables partial application of arguments through calling a function with an
    incomplete set of arguments.

    >>> def mul(x, y):
    ...     return x * y
    >>> mul = curry(mul)

    >>> double = mul(2)
    >>> double(10)
    20

    Also supports keyword arguments

    >>> @curry                  # Can use curry as a decorator
    ... def f(x, y, a=10):
    ...     return a * (x + y)

    >>> add = f(a=1)
    >>> add(2, 3)
    5

    See Also:
        toolz.curried - namespace of curried functions
                        https://toolz.readthedocs.io/en/latest/curry.html
    """

    def __init__(self, *args, **kwargs):
        if not args:
            raise TypeError("__init__() takes at least 2 arguments (1 given)")
        func, args = args[0], args[1:]
        if not callable(func):
            raise TypeError("Input must be callable")

        # curry- or functools.partial-like object?  Unpack and merge arguments
        if (
            hasattr(func, "func")
            and hasattr(func, "args")
            and hasattr(func, "keywords")
            and isinstance(func.args, tuple)
        ):
            _kwargs = {}
            if func.keywords:
                _kwargs.update(func.keywords)
            _kwargs.update(kwargs)
            kwargs = _kwargs
            args = func.args + args
            func = func.func

        if kwargs:
            self._partial = partial(func, *args, **kwargs)
        else:
            self._partial = partial(func, *args)

        self.__doc__ = getattr(func, "__doc__", None)
        self.__name__ = getattr(func, "__name__", "<curry>")
        self.__module__ = getattr(func, "__module__", None)
        self.__qualname__ = getattr(func, "__qualname__", None)
        self._sigspec = None
        self._has_unknown_args = None

    @instanceproperty
    def func(self):
        return self._partial.func

    @instanceproperty
    def __signature__(self):
        sig = inspect.signature(self.func)
        args = self.args or ()
        keywords = self.keywords or {}
        if is_partial_args(self.func, args, keywords, sig) is False:
            raise TypeError("curry object has incorrect arguments")

        params = list(sig.parameters.values())
        skip = 0
        for param in params[: len(args)]:
            if param.kind == param.VAR_POSITIONAL:
                break
            skip += 1

        kwonly = False
        newparams = []
        for param in params[skip:]:
            kind = param.kind
            default = param.default
            if kind == param.VAR_KEYWORD:
                pass
            elif kind == param.VAR_POSITIONAL:
                if kwonly:
                    continue
            elif param.name in keywords:
                default = keywords[param.name]
                kind = param.KEYWORD_ONLY
                kwonly = True
            else:
                if kwonly:
                    kind = param.KEYWORD_ONLY
                if default is param.empty:
                    default = no_default
            newparams.append(param.replace(default=default, kind=kind))

        return sig.replace(parameters=newparams)

    @instanceproperty
    def args(self):
        return self._partial.args

    @instanceproperty
    def keywords(self):
        return self._partial.keywords

    @instanceproperty
    def func_name(self):
        return self.__name__

    def __str__(self):
        return str(self.func)

    def __repr__(self):
        return repr(self.func)

    def __hash__(self):
        return hash(
            (
                self.func,
                self.args,
                frozenset(self.keywords.items()) if self.keywords else None,
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, curry)
            and self.func == other.func
            and self.args == other.args
            and self.keywords == other.keywords
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *args, **kwargs):
        try:
            return self._partial(*args, **kwargs)
        except TypeError as exc:
            if self._should_curry(args, kwargs, exc):
                return self.bind(*args, **kwargs)
            raise

    def _should_curry(self, args, kwargs, exc=None):
        func = self.func
        args = self.args + args
        if self.keywords:
            kwargs = dict(self.keywords, **kwargs)
        if self._sigspec is None:
            sigspec = self._sigspec = _sigs.signature_or_spec(func)
            self._has_unknown_args = has_varargs(func, sigspec) is not False
        else:
            sigspec = self._sigspec

        if is_partial_args(func, args, kwargs, sigspec) is False:
            # Nothing can make the call valid
            return False
        elif self._has_unknown_args:
            # The call may be valid and raised a TypeError, but we curry
            # anyway because the function may have `*args`.  This is useful
            # for decorators with signature `func(*args, **kwargs)`.
            return True
        elif not is_valid_args(func, args, kwargs, sigspec):
            # Adding more arguments may make the call valid
            return True
        else:
            # There was a genuine TypeError
            return False

    def bind(self, *args, **kwargs):
        return type(self)(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        return self._partial(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return curry(self, instance)

    def __reduce__(self):
        func = self.func
        modname = getattr(func, "__module__", None)
        qualname = getattr(func, "__qualname__", None)
        if qualname is None:  # pragma: no cover
            qualname = getattr(func, "__name__", None)
        is_decorated = None
        if modname and qualname:
            if modname == "__main__":
                func = dumps(func)
            else:
                attrs = []
                obj = import_module(modname)
                for attr in qualname.split("."):
                    if isinstance(obj, curry):
                        attrs.append("func")
                        obj = obj.func
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                    attrs.append(attr)
                if isinstance(obj, curry) and obj.func is func:
                    is_decorated = obj is self
                    qualname = ".".join(attrs)
                    func = "%s:%s" % (modname, qualname)

        # functools.partial objects can't be pickled
        userdict = tuple(
            (k, v)
            for k, v in self.__dict__.items()
            if k not in ("_partial", "_sigspec")
        )
        state = (type(self), func, self.args, self.keywords, userdict, is_decorated)
        return _restore_curry, state


def _restore_curry(cls, func, args, kwargs, userdict, is_decorated):
    if isinstance(func, str):
        modname, qualname = func.rsplit(":", 1)
        obj = import_module(modname)
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        if is_decorated:
            return obj
        func = obj.func
    elif isinstance(func, bytes):
        func = loads(func)
    obj = cls(func, *args, **(kwargs or {}))
    obj.__dict__.update(userdict)
    return obj
