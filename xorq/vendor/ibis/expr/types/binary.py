from __future__ import annotations

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr import types as ir

from public import public

import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.expr.types.generic import Column, Scalar, Value


@public
class BinaryValue(Value):
    def hashbytes(
        self,
        how: Literal["md5", "sha1", "sha256", "sha512"] = "sha256",
    ) -> ir.BinaryValue:
        """Compute the binary hash value of `arg`.

        Parameters
        ----------
        how
            Hash algorithm to use

        Returns
        -------
        BinaryValue
            Binary expression
        """
        return ops.HashBytes(self, how).to_expr()


@public
class BinaryScalar(Scalar, BinaryValue):
    pass


@public
class BinaryColumn(Column, BinaryValue):
    pass
