from xorq.catalog.tui.detail.dispatch import register_detail


@register_detail("source")
@register_detail("expr")
@register_detail("composed")
class StandardDetail:
    """Detail strategy for source, expr, and composed entries.

    Shows SQL preview, schema (in/out), and info panel.
    Supports the "run" action.
    """

    def panels(self) -> tuple[str, ...]:
        return ("sql-panel", "schema-panel", "info-panel")

    def update_panels(self, row_data, screen) -> None:
        pass  # CatalogScreen handles rendering for now

    def available_actions(self) -> tuple[str, ...]:
        return ("run",)


@register_detail("unbound_expr")
class UnboundDetail:
    """Detail strategy for unbound expressions.

    Shows SQL preview and schema, but no actions are available --
    unbound expressions must be composed with a source first.
    """

    def panels(self) -> tuple[str, ...]:
        return ("sql-panel", "schema-panel", "info-panel")

    def update_panels(self, row_data, screen) -> None:
        pass  # CatalogScreen handles rendering for now

    def available_actions(self) -> tuple[str, ...]:
        return ()
