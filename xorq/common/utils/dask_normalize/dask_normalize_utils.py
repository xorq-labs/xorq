import pdb
from contextlib import contextmanager
from unittest.mock import (
    Mock,
    patch,
)

import dask
import toolz


@contextmanager
def patch_normalize_token(*typs, f=toolz.functoolz.return_none):
    with patch.dict(
        dask.base.normalize_token._lookup,
        {typ: Mock(side_effect=f) for typ in typs},
    ) as dct:
        mocks = {typ: dct[typ] for typ in typs}
        yield mocks


def normalize_seq_with_caller(*args, caller=""):
    from xorq.common.utils.inspect_utils import (
        get_enclosing_function,
    )

    if caller is None:
        caller = get_enclosing_function(level=2)
    return dask.tokenize._normalize_seq_func(
        (
            caller,
            args,
        )
    )


@toolz.curry
def walk_normalized(f, normalized):
    match normalized:
        case tuple() | list():
            for el in normalized:
                yield from walk_normalized(f, el)
        case str():
            yield f(normalized)
        case bytes():
            yield f(normalized.decode("ascii", errors="replace"))
        case int() | float():
            yield f(normalized)
        case None | slice():
            yield f(normalized)
        case _:
            raise ValueError(
                f"unhandled condition for type {type(normalized)} ({normalized})"
            )


@toolz.curry
def set_trace_on_condition(condition, obj):
    if condition(obj):
        pdb.set_trace()
