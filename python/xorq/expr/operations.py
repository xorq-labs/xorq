"""xorq custom scalar operation nodes."""

from __future__ import annotations

from typing import Any

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.utils.name_utils import tokenize_to_int
from xorq.vendor.ibis.expr.operations.generic import ScalarParameter


_MISSING = object()


class NamedScalarParameter(ScalarParameter):
    """A ScalarParameter with a user-visible semantic label and optional default.

    Use ``xorq.param(name, type)`` to create instances rather than
    constructing this class directly.

    The ``label`` field carries the semantic name supplied by the user (e.g.
    ``"cutoff"``).  The inherited ``name`` property is overridden to return
    ``label`` so that SQL backends use the human-readable name instead of the
    auto-generated ``param_<counter>`` string.

    The ``default`` field holds an optional Python value used when the parameter
    is not explicitly supplied at execution time.  Use the module-level
    ``_MISSING`` sentinel to indicate no default (the parameter is required).
    """

    label: str
    default: Any = _MISSING

    def __init__(self, dtype, label, default=_MISSING, counter=None):
        if counter is None:
            counter = tokenize_to_int(label, dtype)
        if default is not _MISSING:
            if default is not None:
                normalized = dt.dtype(dtype)
                if not dt.infer(default).castable(normalized):
                    raise TypeError(
                        f"Default value {default!r} is not compatible with dtype {normalized}"
                    )
        # Bypass ScalarParameter.__init__ (which does not accept `label`) and
        # delegate directly to the Annotable/Concrete __init__ that sets all
        # declared fields.
        super(ScalarParameter, self).__init__(
            dtype=dtype, counter=counter, label=label, default=default
        )

    @property
    def name(self):
        return self.label
