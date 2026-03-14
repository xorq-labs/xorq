import hashlib
import itertools
import pathlib
import pdb
from contextlib import contextmanager
from unittest.mock import (
    patch,
)

import dask
import toolz


def normalize_attrs(attrs):
    assert hasattr(attrs, "__attrs_attrs__")
    return tuple(sorted(attrs.__getstate__().items()))


@contextmanager
def patch_normalize_token(*typs, f):
    lazy_before = tuple(dask.base.normalize_token._lazy.items())
    try:
        with patch.dict(
            dask.base.normalize_token._lookup,
            values=dict.fromkeys(typs, f),
        ) as dct:
            yield dct
    finally:
        lazy_after = tuple(dask.base.normalize_token._lazy.items())
        registers = tuple(
            register for _, register in set(lazy_before).difference(lazy_after)
        )
        for register in registers:
            register()


def normalize_seq_with_caller(*args, caller=""):
    from xorq.common.utils.inspect_utils import (  # noqa: PLC0415
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


def gen_batches(path, size=2**20):
    with pathlib.Path(path).open("rb") as fh:
        gen = (fh.read(size) for fh in itertools.repeat(fh))
        gen = itertools.takewhile(bool, gen)
        yield from gen


def manual_file_digest(path, digest=hashlib.md5, size=2**20):
    from contextlib import closing  # noqa: PLC0415
    from tarfile import ExFileObject  # noqa: PLC0415

    fh = path if isinstance(path, ExFileObject) else pathlib.Path(path).open("rb")
    with closing(fh):
        obj = digest()
        for chunk in itertools.takewhile(
            bool, (fh.read(size) for fh in itertools.repeat(fh))
        ):
            obj.update(chunk)
        return obj.hexdigest()


def file_digest(path, digest=hashlib.md5, size=2**20):
    from tarfile import ExFileObject  # noqa: PLC0415
    from zipfile import ZipExtFile  # noqa: PLC0415

    if hasattr(hashlib, "file_digest"):
        if isinstance(path, (ExFileObject, ZipExtFile)):
            return hashlib.file_digest(path, digest).hexdigest()
        elif isinstance(path, (str, pathlib.Path)):
            with pathlib.Path(path).open("rb") as fh:
                return hashlib.file_digest(fh, digest).hexdigest()
        else:
            raise ValueError(f"Don't know how to handle type {type}")
    else:
        # python 3.10
        return manual_file_digest(path, digest, size=size)


def normalize_read_path_md5sum(path):
    tpls = (("content-md5sum", file_digest(path)),)
    return tpls


@contextmanager
def patch_normalize_op_caching():
    import functools  # noqa: PLC0415

    import xorq.common.utils.dask_normalize.dask_normalize_expr as mod  # noqa: PLC0415

    attr = "normalize_op"
    cached_attr = functools.cache(getattr(mod, attr))
    with patch.object(mod, attr, cached_attr):
        yield


def patched_tokenize(expr):
    with patch_normalize_op_caching():
        return dask.base.tokenize(expr)
