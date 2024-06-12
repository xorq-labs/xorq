import dask
import toolz


from letsql.common.utils.logging_utils import get_logger


logger = get_logger(__name__)


none_tokenized = "8c9f081a88f539969f3dff99d6e05e36"


@toolz.curry
def maybe_hotfix(obj, attrname, target_tokenized, hotfix):
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
    if tokenized == target_tokenized:
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
