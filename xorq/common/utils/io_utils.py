from contextlib import contextmanager
from io import TextIOWrapper


@contextmanager
def maybe_open(obj, *args, **kwargs):
    if isinstance(obj, TextIOWrapper):
        yield obj
    else:
        with open(obj, *args, **kwargs) as fh:
            yield fh
