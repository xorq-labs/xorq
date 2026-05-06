from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from typing import Any, TypeVar

import toolz


_T = TypeVar("_T")

count = itertools.count()


def return_constant(value: _T) -> Callable[..., _T]:
    def wrapped(*args: Any, **kwargs: Any) -> _T:
        return value

    return wrapped


@toolz.curry
def log_excepts(
    f: Callable[..., Any], exception: type[Exception] = Exception
) -> Callable[..., Any]:
    # file logger
    # from xorq.common.utils.logging_utils import get_logger

    # print logger
    from structlog import get_logger  # noqa: PLC0415

    logger = get_logger(__name__)

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        i = next(count)
        try:
            logger.info(f"{f.__name__} :: entering :: {i}")
            value = f(*args, **kwargs)
            logger.info(f"{f.__name__} :: exiting  :: {i}")
            return value
        except exception:
            logger.exception("exception!")

    return wrapper


@toolz.curry
def maybe_log_excepts(
    f: Callable[..., Any],
    exception: type[Exception] = Exception,
    debug: bool | None = None,
) -> Callable[..., Any]:
    from xorq.config import options  # noqa: PLC0415

    if options.debug or debug:
        return log_excepts(f, exception=exception)
    else:
        return f


@toolz.curry
def with_lock(lock: Any, f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return f(*args, **kwargs)

    return wrapper


@toolz.curry
def if_not_none(f: Callable[[_T], Any], value: _T | None) -> Any:
    return value if value is None else f(value)
