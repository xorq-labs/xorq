from typing import Protocol, runtime_checkable


@runtime_checkable
class DetailStrategy(Protocol):
    """Strategy for populating right-panel content based on ExprKind.

    Each ExprKind value is registered with a DetailStrategy that controls:
    - Which panels are visible when an entry of that kind is highlighted
    - How to populate those panels with data from the entry
    - What actions (run, sink, serve, train) are available for that kind
    """

    def panels(self) -> tuple[str, ...]:
        """Return panel IDs that should be visible for this kind.

        Example: ("sql-panel", "schema-panel", "info-panel")
        """
        ...

    def update_panels(self, row_data, screen) -> None:
        """Populate visible panels with data from row_data.

        screen is the CatalogScreen instance, providing access to
        query_one() for widget lookups.
        """
        ...

    def available_actions(self) -> tuple[str, ...]:
        """Return action types available for this kind.

        Example: ("run",) or ("run", "sink", "serve")
        """
        ...
