from xorq_datafusion._internal import parser


def __getattr__(name):
    return getattr(parser, name)
