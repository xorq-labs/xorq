import functools
import logging
import pathlib
import time

from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
)
from xorq.config import options


logger = logging.getLogger(__name__)

whitelist = [
    "astronauts",
    "awards_players",
    "batting",
    "diamonds",
    "functional_alltypes",
    "iris",
    "penguins",
    "hn_posts_nano",
    "hn-data-small.parquet",
]

_PIN_META_RETRIES = 3
_PIN_META_RETRY_DELAY = 1.0


def _pin_meta_with_retry(board, name):
    last_exc = None
    for attempt in range(_PIN_META_RETRIES):
        try:
            meta = board.pin_meta(name)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "pin_meta(%r) raised %s (attempt %d/%d)",
                name,
                exc,
                attempt + 1,
                _PIN_META_RETRIES,
                exc_info=True,
            )
        else:
            if meta is not None:
                return meta
            logger.warning(
                "pin_meta(%r) returned None (attempt %d/%d)",
                name,
                attempt + 1,
                _PIN_META_RETRIES,
            )
        if attempt < _PIN_META_RETRIES - 1:
            delay = _PIN_META_RETRY_DELAY * (2**attempt)
            time.sleep(delay)
    raise RuntimeError(
        f"failed to fetch pin metadata for {name!r} after {_PIN_META_RETRIES} attempts"
    ) from last_exc


@functools.cache
def get_name_to_suffix() -> dict[str, str]:
    board = options.pins.get_board()
    dct = {
        name: pathlib.Path(_pin_meta_with_retry(board, name).file).suffix
        for name in board.pin_list()
        if name in whitelist
    }
    return dct


def get_table_from_name(name, backend, table_name=None, deferred=True, **kwargs):
    suffix = get_name_to_suffix().get(name)
    match suffix:
        case ".parquet":
            if deferred:
                method = functools.partial(deferred_read_parquet, con=backend)
            else:
                method = backend.read_parquet
        case ".csv":
            if deferred:
                method = functools.partial(deferred_read_csv, con=backend)
            else:
                method = backend.read_csv
        case _:
            raise ValueError(
                f"unsupported file suffix {suffix!r}; expected .parquet or .csv"
            )
    path = options.pins.get_path(name)
    return method(path, table_name=table_name or name, **kwargs)
