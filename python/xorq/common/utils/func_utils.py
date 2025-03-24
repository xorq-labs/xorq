import functools
import itertools

import toolz

import xorq as xo


count = itertools.count()


def return_constant(value):
    def wrapped(*args, **kwargs):
        return value

    return wrapped


@toolz.curry
def log_excepts(f, exception=Exception):
    # file logger
    # from xorq.common.utils.logging_utils import get_logger

    # print logger
    from structlog import get_logger

    logger = get_logger(__name__)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
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
def maybe_log_excepts(f, exception=Exception, debug=None):
    if xo.options.debug or debug:
        return log_excepts(f, exception=exception)
    else:
        return f


@toolz.curry
def with_lock(lock, f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with lock:
            return f(*args, **kwargs)

    return wrapper
