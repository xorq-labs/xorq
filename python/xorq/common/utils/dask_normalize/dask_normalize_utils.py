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


def normalize_seq_with_caller(*args):
    # from xorq.common.utils.inspect_utils import get_enclosing_function
    # # FIXME: can we make this quicker?
    # # # if not: either hardcode the caller name or get caller conditional on debug value
    # caller = get_enclosing_function(level=2)
    caller = ""
    return dask.tokenize._normalize_seq_func(
        (
            caller,
            args,
        )
    )
