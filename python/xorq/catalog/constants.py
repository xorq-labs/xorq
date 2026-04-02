try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class CatalogInfix(StrEnum):
    ALIAS = "aliases"
    ENTRY = "entries"
    METADATA = "metadata"


METADATA_APPEND = ".metadata.yaml"
VALID_SUFFIXES = ((PREFERRED_SUFFIX := ".zip"),)
CATALOG_YAML_NAME = "catalog.yaml"

MAIN_BRANCH = "main"
ANNEX_BRANCH = "git-annex"
DEFAULT_REMOTE = "origin"
