from xorq_datafusion._internal import expr


def __getattr__(name):
    return getattr(expr, name)
