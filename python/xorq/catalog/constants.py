try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class CatalogInfix(StrEnum):
    ALIAS = "aliases"
    ENTRY = "entries"
    METADATA = "metadata"


CATALOG_REMOTE_KEY = "remote"


METADATA_APPEND = ".metadata.yaml"
VALID_SUFFIXES = ((PREFERRED_SUFFIX := ".zip"),)
CATALOG_YAML_NAME = "catalog.yaml"
