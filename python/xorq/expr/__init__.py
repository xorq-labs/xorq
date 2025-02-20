from xorq._internal import expr


def __getattr__(name):
    return getattr(expr, name)
