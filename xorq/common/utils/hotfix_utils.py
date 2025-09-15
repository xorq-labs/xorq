import warnings

import dask
import toolz


try:
    import cityhash  # noqa: F401
except ImportError:
    warnings.warn(
        "cityhash is not installed, some functionality will not work", UserWarning
    )

from xorq.common.utils.logging_utils import get_logger


logger = get_logger(__name__)


none_tokenized = dask.base.tokenize(None)


@toolz.curry
def hotfix(obj, attrname, target_tokenized, hotfix):
    return maybe_hotfix(obj, attrname, target_tokenized, hotfix, definitely=True)


@toolz.curry
def maybe_hotfix(obj, attrname, target_tokenized, hotfix, definitely=False):
    to_hotfix = getattr(obj, attrname, None)
    tokenized = dask.base.tokenize(to_hotfix)
    dct = {
        "obj.__name__": obj.__name__,
        "obj.__module__": obj.__module__,
        "attrname": attrname,
        "tokenized": tokenized,
        "target_tokenized": target_tokenized,
        "hotfix_tokenized": dask.base.tokenize(hotfix),
    }

    if definitely or (tokenized == target_tokenized):
        if not isinstance(hotfix, property):
            setattr(hotfix, "_original", to_hotfix)
        else:
            if tokenized != none_tokenized:
                raise ValueError("Don't know how to retain _original")
        setattr(obj, attrname, hotfix)
        logger.info("hotfixing", **dct)
    else:
        logger.info("not hotfixing", **dct)
    return hotfix
