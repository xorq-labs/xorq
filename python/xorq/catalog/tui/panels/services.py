from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable

from xorq.catalog.tui.models import SERVICE_COLUMNS, ServiceStatus


class ServicesPanel(Vertical):
    """Panel showing active Flight gRPC servers.

    Displays a DataTable of running/stopped services with status,
    name, endpoint, and start time. Services are managed externally
    and added/updated/removed through public methods.
    """

    DEFAULT_CSS = """
    ServicesPanel {
        height: 1fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        border-subtitle-color: #C1F0FF;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._services: dict[str, ServiceStatus] = {}

    def compose(self) -> ComposeResult:
        yield DataTable(id="services-table")

    def on_mount(self) -> None:
        table = self.query_one("#services-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for col in SERVICE_COLUMNS:
            table.add_column(col, key=col)
        self.border_title = "Services"
        self.border_subtitle = "none"

    def add_service(self, status: ServiceStatus) -> None:
        self._services[status.name] = status
        self._render()

    def update_service(self, name: str, new_status: str) -> None:
        match self._services.get(name):
            case None:
                return
            case svc:
                self._services[name] = ServiceStatus(
                    name=svc.name,
                    endpoint=svc.endpoint,
                    status=new_status,
                    started_at=svc.started_at,
                )
        self._render()

    def remove_service(self, name: str) -> None:
        self._services.pop(name, None)
        self._render()

    def _render(self) -> None:
        table = self.query_one("#services-table", DataTable)
        table.clear()
        for i, svc in enumerate(self._services.values()):
            table.add_row(*svc.row, key=str(i))
        match len(self._services):
            case 0:
                self.border_subtitle = "none"
            case n:
                running = sum(
                    1 for s in self._services.values() if s.status == "running"
                )
                self.border_subtitle = f"{running}/{n} running"
