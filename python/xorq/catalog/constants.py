try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class CatalogInfix(StrEnum):
    ALIAS = "aliases"
    ENTRY = "entries"
    METADATA = "metadata"


METADATA_APPEND = ".metadata.yaml"
VALID_SUFFIXES = (".tar.gz", (PREFERRED_SUFFIX := ".tgz"))
CATALOG_YAML_NAME = "catalog.yaml"
