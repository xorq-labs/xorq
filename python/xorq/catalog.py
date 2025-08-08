import os
import uuid
import yaml
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import xorq as xo

# Default catalog file path, using XDG_CONFIG_HOME or ~/.config/xorq/catalog.yaml
DEFAULT_CATALOG_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "xorq" / "catalog.yaml"

def get_catalog_path(path=None):
    if path:
        return Path(path)
    return DEFAULT_CATALOG_PATH

def load_catalog(path=None):
    """
    Load the catalog from the given path, or initialize a new catalog if it does not exist.
    """
    catalog_path = get_catalog_path(path)
    if not catalog_path.exists():
        now = datetime.now(timezone.utc).isoformat()
        return {
            "apiVersion": "xorq.dev/v1",
            "kind": "XorqCatalog",
            "metadata": {
                "catalog_id": str(uuid.uuid4()),
                "created_at": now,
                "updated_at": now,
                "tool_version": xo.__version__,
            },
            "aliases": {},
            "entries": [],
        }
    with catalog_path.open() as f:
        return yaml.safe_load(f) or {}

def save_catalog(catalog, path=None):
    """
    Save the catalog to the given path.
    """
    catalog_path = get_catalog_path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    with catalog_path.open("w") as f:
        yaml.safe_dump(catalog, f, sort_keys=False)