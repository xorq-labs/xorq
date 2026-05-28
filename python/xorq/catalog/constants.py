from pathlib import Path


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
POINTER_SUFFIX = ".pointer"
CATALOG_YAML_NAME = "catalog.yaml"
CONTENT_STORE_YAML = "content_store.yaml"

MAIN_BRANCH = "main"
ANNEX_BRANCH = "git-annex"
DEFAULT_REMOTE = "origin"

DEFAULT_CATALOG_NAME = "default"
DEFAULT_CATALOG_CONFIG = Path("~/.config/xorq/catalog-default").expanduser()
