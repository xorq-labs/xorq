"""APIs for creating user-defined functions."""

from __future__ import annotations

from xorq.vendor.ibis.legacy.udf.vectorized import analytic, elementwise, reduction


class udf:
    @staticmethod
    def elementwise(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.elementwise."""

        return elementwise(input_type, output_type)

    @staticmethod
    def reduction(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.reduction."""
        return reduction(input_type, output_type)

    @staticmethod
    def analytic(input_type, output_type):
        """Alias for ibis.legacy.udf.vectorized.analytic."""
        return analytic(input_type, output_type)
