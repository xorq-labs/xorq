from __future__ import annotations

from pathlib import Path


METADATA_APPEND = ".metadata.yaml"
VALID_SUFFIXES = ((PREFERRED_SUFFIX := ".zip"),)
POINTER_SUFFIX = ".pointer"
CATALOG_YAML_NAME = "catalog.yaml"
CONTENT_STORE_YAML = "content_store.yaml"
# The `build_metadata.json` key, and the sidecar key, naming a rebased entry's
# ancestor.
REBASED_FROM = "rebased_from"

MAIN_BRANCH = "main"
ANNEX_BRANCH = "git-annex"
DEFAULT_REMOTE = "origin"

DEFAULT_CATALOG_NAME = "default"
DEFAULT_CATALOG_CONFIG = Path("~/.config/xorq/catalog-default").expanduser()
