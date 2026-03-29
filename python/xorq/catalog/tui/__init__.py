"""xorq catalog TUI package.

Re-exports all public symbols for backward compatibility with
``from xorq.catalog.tui import CatalogTUI, CatalogScreen, ...``.
"""

from xorq.catalog.tui.app import XORQ_DARK, CatalogTUI
from xorq.catalog.tui.models import (  # noqa: F401
    CACHE_PANEL_COLUMNS,
    CACHE_TYPE_LABELS,
    COLUMNS,
    DEFAULT_REFRESH_INTERVAL,
    GIT_LOG_COLUMNS,
    REVISION_COLUMNS,
    RUN_COLUMNS,
    SCHEMA_PREVIEW_COLUMNS,
    SERVICE_COLUMNS,
    CacheRowData,
    CatalogRowData,
    GitLogRowData,
    RevisionRowData,
    RunConfig,
    RunRowData,
    ServiceStatus,
    SinkConfig,
    _build_alias_multimap,
    _build_cache_entry_map,
    _build_cache_rows,
    _build_git_log_rows,
    _build_run_rows,
    _cache_type_display,
    _catalog_aliases_cached,
    _catalog_list_cached,
    _compute_duration,
    _entry_info,
    _format_cached,
    _format_run_date,
    _format_run_detail,
    _format_size,
    _get_catalog_aliases,
    _get_catalog_list,
    _load_catalog_row,
    _parquet_to_cache_row,
    _render_sql_dag,
    _revision_pair,
    _run_to_row,
    _TogglePanelState,
)
from xorq.catalog.tui.screens.catalog import CatalogScreen, XorqSQLStyle
from xorq.catalog.tui.screens.modals import RunOptionsScreen
from xorq.catalog.tui.screens.run_data import RunDataScreen


__all__ = [
    "CACHE_PANEL_COLUMNS",
    "CACHE_TYPE_LABELS",
    "COLUMNS",
    "CacheRowData",
    "CatalogRowData",
    "CatalogScreen",
    "CatalogTUI",
    "DEFAULT_REFRESH_INTERVAL",
    "GIT_LOG_COLUMNS",
    "GitLogRowData",
    "REVISION_COLUMNS",
    "RUN_COLUMNS",
    "RevisionRowData",
    "RunConfig",
    "RunDataScreen",
    "RunOptionsScreen",
    "RunRowData",
    "SCHEMA_PREVIEW_COLUMNS",
    "SERVICE_COLUMNS",
    "ServiceStatus",
    "SinkConfig",
    "XORQ_DARK",
    "XorqSQLStyle",
]
