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

    @classmethod
    def from_expr(cls, expr, catalog):
        """Recover an ExprComposer from a tagged expression.

        Walks the HashingTag nodes embedded by a prior ``ExprComposer.expr``
        call and reconstructs the original ``source``, ``transforms``,
        ``code``, and ``alias`` fields.

        Parameters
        ----------
        expr : Expr
            An expression previously produced by ``ExprComposer.expr``.
        catalog : Catalog
            The catalog that owns the referenced entries.
        """
        from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
        from xorq.expr.relations import HashingTag  # noqa: PLC0415

        # walk_nodes returns outermost-first; reverse to get composition order
        # reversed order: SOURCE, transforms..., CODE (if present)
        nodes = tuple(
            reversed(
                tuple(
                    ht
                    for ht in (walk_nodes(HashingTag, expr) or ())
                    if ht.metadata.get("tag") in frozenset(CatalogTag)
                )
            )
        )
        if not nodes or not nodes[0].metadata["tag"] == CatalogTag.SOURCE:
            raise ValueError(
                "No catalog-source tag found; expression was not produced by ExprComposer"
            )

        if nodes[-1].metadata["tag"] == CatalogTag.CODE:
            (*nodes, code_node) = nodes
            code = code_node.metadata["code"]
        else:
            code = None

        (source_node, *transform_nodes) = nodes
        if not all(n.metadata["tag"] == CatalogTag.TRANSFORM for n in transform_nodes):
            raise ValueError(
                "Unexpected non-transform tag found between source and code tags"
            )

        (source_entry, *transform_entries) = (
            catalog.get_catalog_entry(node.metadata["entry_name"]) for node in nodes
        )
        alias = source_node.metadata.get("alias")
        return cls(
            source=source_entry,
            transforms=tuple(transform_entries),
            code=code,
            alias=alias,
        )
