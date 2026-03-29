from xorq.catalog.tui.detail.dispatch import register_detail


@register_detail("ml_model")
class MLModelDetail:
    """Detail strategy for ML model entries.

    Will show feature importance and model metrics panels.
    Stub: falls back to standard panel set until ML panel widgets exist.
    """

    def panels(self) -> tuple[str, ...]:
        return ("sql-panel", "schema-panel", "info-panel")

    def update_panels(self, row_data, screen) -> None:
        pass

    def available_actions(self) -> tuple[str, ...]:
        return ("run",)
