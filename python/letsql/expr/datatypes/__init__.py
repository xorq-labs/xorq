from letsql.vendor.ibis.expr.datatypes import String


class LargeString(String):
    """A type representing a large_string.

    Notes
    -----
    Because of differences in the way different backends handle strings, we
    cannot assume that strings are UTF-8 encoded.

    """

    scalar = "StringScalar"
    column = "StringColumn"
