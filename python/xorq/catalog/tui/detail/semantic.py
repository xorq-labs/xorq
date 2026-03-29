from xorq.catalog.tui.detail.dispatch import register_detail


@register_detail("semantic_model")
class SemanticModelDetail:
    """Detail strategy for semantic model entries.

    Will show metrics/dimensions panel instead of SQL.
    Stub: falls back to standard panel set until semantic panel widgets exist.
    """

    def panels(self) -> tuple[str, ...]:
        return ("sql-panel", "schema-panel", "info-panel")

    def update_panels(self, row_data, screen) -> None:
        pass

    def available_actions(self) -> tuple[str, ...]:
        return ("run",)
