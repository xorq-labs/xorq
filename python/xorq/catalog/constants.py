from __future__ import annotations

from pathlib import Path


METADATA_APPEND = ".metadata.yaml"
VALID_SUFFIXES = ((PREFERRED_SUFFIX := ".zip"),)
CATALOG_YAML_NAME = "catalog.yaml"

MAIN_BRANCH = "main"
ANNEX_BRANCH = "git-annex"
DEFAULT_REMOTE = "origin"

DEFAULT_CATALOG_NAME = "default"
DEFAULT_CATALOG_CONFIG = Path("~/.config/xorq/catalog-default").expanduser()
