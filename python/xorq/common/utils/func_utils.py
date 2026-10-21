import contextlib
import functools
import inspect
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

    @contextlib.contextmanager
    def logged():
        i = next(count)
        try:
            logger.info(f"{f.__name__} :: entering :: {i}")
            yield
            logger.info(f"{f.__name__} :: exiting  :: {i}")
        except exception:
            # adds logging, not semantics: `maybe_log_excepts` leaves `f`
            # undecorated when debug is off, so swallowing here lost errors
            # that plain mode surfaces
            logger.exception("exception!")
            raise

    # a generator's body runs on iteration, after the call has returned
    # (Flight's do_action), so the logging has to wrap the iteration
    @functools.wraps(f)
    def gen_wrapper(*args, **kwargs):
        with logged():
            yield from f(*args, **kwargs)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with logged():
            return f(*args, **kwargs)

    return gen_wrapper if inspect.isgeneratorfunction(f) else wrapper


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
