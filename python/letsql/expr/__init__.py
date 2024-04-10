from letsql._internal import expr


def __getattr__(name):
    return getattr(expr, name)
