import functools
import itertools

import toolz


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
    from structlog import get_logger  # noqa: PLC0415

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
            # log where it happened, then propagate. Swallowing here made debug
            # mode *lose* errors that plain mode surfaces: `maybe_log_excepts`
            # leaves `f` undecorated when debug is off, so returning None turned
            # a failed Flight RPC into a clean, empty, cacheable stream under
            # XORQ_DEBUG=1 alone. This decorator adds logging, not semantics.
            logger.exception("exception!")
            raise

    return wrapper


@toolz.curry
def maybe_log_excepts(f, exception=Exception, debug=None):
    from xorq.config import options  # noqa: PLC0415

    if options.debug or debug:
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


@toolz.curry
def if_not_none(f, value):
    return value if value is None else f(value)
