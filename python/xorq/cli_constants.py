try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class OutputFormats(StrEnum):
    csv = "csv"
    json = "json"
    parquet = "parquet"
    arrow = "arrow"


DEFAULT_OUTPUT_FORMAT = OutputFormats.parquet
DEFAULT_CACHE_TYPE = "modification-time"
