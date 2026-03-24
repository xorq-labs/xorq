from functools import cached_property

from attr import Attribute, field, frozen
from attr.validators import deep_iterable, instance_of, optional

from xorq.catalog.bind import (
    CatalogTag,
    _eval_code,
    _make_source_expr,
    _validate_one_catalog,
    bind,
)
from xorq.catalog.catalog import CatalogEntry


def _same_catalog(instance, attribute: Attribute, value):
    if value:
        _validate_one_catalog(instance.source, value)


@frozen
class ExprComposer:
    """A recipe for composing catalog entries into an expression.

    Accepts a source entry with optional transforms and/or inline code.
    A bare source (no transforms, no code) produces the source expression
    directly.  Provenance tags (``HashingTag``) are applied by ``bind()``
    and ``_resolve_source`` automatically — this class adds code-step
    tagging on top.
    """

    source = field(validator=instance_of(CatalogEntry))
    transforms = field(
        factory=tuple,
        converter=tuple,
        validator=[deep_iterable(instance_of(CatalogEntry)), _same_catalog],
    )
    code = field(default=None, validator=optional(instance_of(str)))
    alias = field(default=None, validator=optional(instance_of(str)))

    @cached_property
    def expr(self):
        if self.transforms:
            current = bind(self.source, *self.transforms, alias=self.alias)
        else:
            current = _make_source_expr(self.source, alias=self.alias)

        if self.code is not None:
            current = _eval_code(self.code, current)
            current = current.hashing_tag(CatalogTag.CODE, code=self.code)

        return current
