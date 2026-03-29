from textual import work
from textual.app import App
from textual.theme import Theme

from xorq.catalog.tui.models import DEFAULT_REFRESH_INTERVAL
from xorq.catalog.tui.screens.catalog import CatalogScreen


XORQ_DARK = Theme(
    name="xorq-dark",
    primary="#C1F0FF",
    secondary="#4AA8EC",
    warning="#F5CA2C",
    error="#FF4757",
    success="#2BBE75",
    accent="#C1F0FF",
    foreground="#C1F0FF",
    background="#05181A",
    surface="#0a2a2e",
    panel="#0f3338",
    dark=True,
)


class CatalogTUI(App):
    TITLE = "xorq catalog"
    CSS = """
    #main-split {
        height: 1fr;
    }
    #left-column {
        width: 2fr;
    }
    #right-column {
        width: 3fr;
    }
    #catalog-panel {
        height: 2fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        background: $surface;
    }
    #catalog-table {
        height: 1fr;
    }
    #runs-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
        border-subtitle-color: #F5CA2C;
    }
    #runs-table {
        height: 1fr;
    }
    #revisions-panel {
        height: 1fr;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        border-subtitle-color: #5abfb5;
    }
    #revisions-preview-table {
        height: 1fr;
    }
    #git-log-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
    }
    #git-log-table {
        height: 1fr;
    }
    #main-content-panel {
        height: 2fr;
    }
    #sql-panel {
        height: 1fr;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
        border-subtitle-color: #2BBE75;
    }
    #sql-panel:focus-within {
        border: double #2BBE75;
    }
    #sql-preview {
        height: auto;
        padding: 1 2;
    }
    #inline-data-panel {
        height: 1fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        border-subtitle-color: #C1F0FF;
    }
    #inline-data-status {
        height: 1;
        padding: 0 1;
    }
    #inline-data-table {
        height: 1fr;
    }
    DataTable:focus {
        border: none;
    }
    #info-panel {
        height: auto;
        max-height: 6;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        padding: 0 1;
    }
    #info-content {
        height: auto;
    }
    #schema-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
        border-subtitle-color: #4AA8EC;
    }
    #schema-split {
        height: 1fr;
    }
    #schema-in-half {
        width: 1fr;
    }
    #schema-out-half {
        width: 1fr;
    }
    #schema-in-table {
        height: 1fr;
    }
    #schema-preview-table {
        height: 1fr;
    }
    #caches-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
        border-subtitle-color: #F5CA2C;
    }
    #caches-table {
        height: 1fr;
    }
    #data-preview-panel {
        height: 2fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        border-subtitle-color: #C1F0FF;
    }
    #data-preview-status {
        height: 1;
        padding: 0 2;
    }
    #data-preview-table {
        height: 1fr;
    }
    #profiles-panel {
        height: 1fr;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
    }
    #profiles-table {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    RunDataScreen {
        background: $surface;
    }
    RunDataScreen #run-data-container {
        height: 1fr;
    }
    RunDataScreen #run-data-status {
        height: 1;
        padding: 0 2;
        background: $panel;
    }
    RunDataScreen #run-data-table {
        height: 1fr;
    }
    RunOptionsScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    RunOptionsScreen #run-options-container {
        width: 40;
        height: auto;
        max-height: 14;
        border: solid #5abfb5;
        background: $surface;
        padding: 0 1;
    }
    RunOptionsScreen #run-options-title {
        height: 1;
        color: #5abfb5;
    }
    RunOptionsScreen #cache-strategy {
        height: auto;
        margin: 0;
    }
    RunOptionsScreen #ttl-row {
        height: 3;
        padding: 0 1;
    }
    RunOptionsScreen #ttl-label {
        width: auto;
    }
    RunOptionsScreen #ttl-input {
        width: 10;
    }
    #services-panel {
        height: 1fr;
    }
    SinkOptionsScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    SinkOptionsScreen #sink-options-container {
        width: 50;
        height: auto;
        max-height: 22;
        border: solid #5abfb5;
        background: $surface;
        padding: 0 1;
    }
    SinkOptionsScreen #sink-options-title {
        height: 1;
        color: #5abfb5;
    }
    SinkOptionsScreen #sink-strategy {
        height: auto;
        margin: 0;
    }
    ServeOptionsScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    ServeOptionsScreen #serve-options-container {
        width: 40;
        height: auto;
        max-height: 14;
        border: solid #5abfb5;
        background: $surface;
        padding: 0 1;
    }
    ServeOptionsScreen #serve-options-title {
        height: 1;
        color: #5abfb5;
    }
    ActionChooserScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    ActionChooserScreen #action-chooser-container {
        width: 30;
        height: auto;
        max-height: 10;
        border: solid #5abfb5;
        background: $surface;
        padding: 0 1;
    }
    ActionChooserScreen #action-chooser-title {
        height: 1;
        color: #5abfb5;
    }
    ActionChooserScreen #action-choices {
        height: auto;
        margin: 0;
    }
    TelemetryScreen {
        background: $surface;
    }
    TelemetryScreen #telemetry-split {
        height: 1fr;
    }
    TelemetryScreen #telemetry-spans-panel {
        width: 3fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
    }
    TelemetryScreen #telemetry-spans-title {
        height: 1;
        color: #F5CA2C;
    }
    TelemetryScreen #telemetry-tree {
        height: 1fr;
    }
    TelemetryScreen #telemetry-meta-panel {
        width: 2fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
    }
    TelemetryScreen #telemetry-meta-title {
        height: 1;
        color: #4AA8EC;
    }
    TelemetryScreen #telemetry-meta-table {
        height: 1fr;
    }
    ComposeScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    ComposeScreen #compose-container {
        width: 90;
        height: auto;
        max-height: 34;
        border: solid #2BBE75;
        background: $surface;
        padding: 0 1;
    }
    ComposeScreen #compose-title {
        height: 1;
        color: #2BBE75;
    }
    ComposeScreen #compose-body {
        height: auto;
        max-height: 30;
    }
    ComposeScreen #compose-left {
        width: 2fr;
    }
    ComposeScreen #compose-right {
        width: 3fr;
    }
    ComposeScreen #entries-label {
        height: 1;
    }
    ComposeScreen #compose-entries-table {
        height: 1fr;
        max-height: 20;
    }
    ComposeScreen #chain-label {
        height: 1;
        color: #2BBE75;
    }
    ComposeScreen #compose-chain-table {
        height: auto;
        max-height: 8;
    }
    ComposeScreen #compose-validation {
        height: auto;
        max-height: 3;
        padding: 0 1;
    }
    ComposeScreen #code-label {
        height: 1;
    }
    ComposeScreen #compose-code {
        height: auto;
        min-height: 4;
        max-height: 10;
    }
    ComposeScreen #compose-alias-row {
        height: 3;
        padding: 0 1;
    }
    ComposeScreen #alias-label {
        width: auto;
    }
    ComposeScreen #compose-alias-input {
        width: 1fr;
    }
    """

    def __init__(self, make_catalog, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._catalog = None
        self._make_catalog = make_catalog
        self._refresh_interval = refresh_interval
        self.register_theme(XORQ_DARK)
        self.theme = "xorq-dark"

    def on_mount(self) -> None:
        self.push_screen(CatalogScreen(refresh_interval=self._refresh_interval))
        self._load_catalog()

    @work(thread=True)
    def _load_catalog(self) -> None:
        catalog = self._make_catalog()
        self.app.call_from_thread(self._set_catalog, catalog)

    def _set_catalog(self, catalog) -> None:
        self._catalog = catalog
        self.screen._do_refresh()
