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

# Exporter, directory diff utilities, and target resolution for catalog diff
import json
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

def normalize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_obj(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [normalize_obj(v) for v in obj]
    return obj

def dump_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)

def dump_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def build_virtual_export_tree(entry: Dict[str, Any], rev: Dict[str, Any]) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    index = {
        "alias": None,
        "entry_id": entry.get("entry_id"),
        "revision_id": rev.get("revision_id"),
        "build_id": (rev.get("build") or {}).get("build_id"),
        "expr_digest": (rev.get("expr_hashes") or {}).get("expr"),
        "meta_digest": rev.get("meta_digest"),
    }
    files["EXPORT_INDEX.yaml"] = dump_yaml(normalize_obj(index)).encode()
    node_hashes = rev.get("node_hashes") or []
    files["expr/expr.yaml"] = dump_yaml({"root_expr": index["expr_digest"]}).encode()
    for nh in node_hashes:
        files[f"expr/nodes/{nh}.yaml"] = dump_yaml({"id": nh}).encode()
    # Schema fingerprint (if available)
    fp = rev.get("metadata", {}).get("schema_fingerprint") or rev.get("schema_fingerprint")
    if fp:
        files["schema/output.schema.json"] = dump_json({"fingerprint": fp}).encode()
    reads = rev.get("metadata", {}).get("read_set", [])
    if reads:
        files["reads/sources.yaml"] = dump_yaml(normalize_obj(reads)).encode()
    files["meta/volatile.yaml"] = dump_yaml({"created_at": rev.get("created_at")} ).encode()
    return files

def write_tree(root: Path, files: Dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

def read_text(path: Path) -> List[str]:
    return path.read_text().splitlines(keepends=True)

def unified_dir_diff(left_root: Path, right_root: Path) -> Tuple[bool, str]:
    left_files = sorted([p for p in left_root.rglob("*") if p.is_file()])
    right_files = sorted([p for p in right_root.rglob("*") if p.is_file()])
    left_map = {str(p.relative_to(left_root)): p for p in left_files}
    right_map = {str(p.relative_to(right_root)): p for p in right_files}
    all_keys = sorted(set(left_map) | set(right_map))
    chunks: List[str] = []
    different = False
    for rel in all_keys:
        lp = left_map.get(rel)
        rp = right_map.get(rel)
        if lp and rp:
            if lp.read_bytes() == rp.read_bytes():
                continue
            different = True
            diff = difflib.unified_diff(
                read_text(lp), read_text(rp),
                fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""
            )
            chunks.extend(diff)
            chunks.append("\n")
        elif lp and not rp:
            different = True
            chunks.append(f"--- a/{rel}\n+++ /dev/null\n")
        else:
            different = True
            chunks.append(f"--- /dev/null\n+++ b/{rel}\n")
    return different, "".join(chunks)

@dataclass
class Target:
    entry_id: str
    rev: Optional[str]
    alias: bool

def resolve_target(target: str, catalog: Dict[str, Any]) -> Target:
    if "@" in target:
        base, rev = target.split("@", 1)
    else:
        base, rev = target, None
    aliases = catalog.get("aliases", {}) or {}
    if base in aliases:
        entry_id = aliases[base]["entry_id"]
        alias_flag = rev is None
    else:
        entry_ids = [e.get("entry_id") for e in catalog.get("entries", [])]
        if base not in entry_ids:
            raise ValueError(f"Unknown target: {target}")
        entry_id = base
        alias_flag = False
    return Target(entry_id=entry_id, rev=rev, alias=alias_flag)

def resolve_build_dir(token: str, catalog: Dict[str, Any]) -> Path:
    """
    Resolve a build directory from various token types:
      - existing directory path
      - alias or alias@rev
      - entry_id or entry_id@rev
      - build_id
    """
    p = Path(token)
    if p.exists() and p.is_dir():
        return p
    # Alias or entry@rev or entry
    if "@" in token:
        base, rev = token.split("@", 1)
    else:
        base, rev = token, None
    aliases = catalog.get("aliases", {}) or {}
    entries = catalog.get("entries", []) or []
    # Alias case
    if base in aliases:
        mapping = aliases[base]
        entry_id = mapping["entry_id"]
        revision = rev or mapping.get("revision_id")
    else:
        # Entry ID case
        entry_ids = [e.get("entry_id") for e in entries]
        if base in entry_ids:
            entry_id = base
            entry = next(e for e in entries if e.get("entry_id") == entry_id)
            revision = rev or entry.get("current_revision")
        else:
            # Build ID case
            for e in entries:
                for r in e.get("history", []):
                    if r.get("build", {}).get("build_id") == token:
                        path = r.get("build", {}).get("path")
                        if not path:
                            raise ValueError(f"No build path for build_id: {token}")
                        return Path(path)
            raise ValueError(f"Unknown build target: {token}")
    # Lookup entry and revision
    entry = next(e for e in entries if e.get("entry_id") == entry_id)
    rev_obj = next((r for r in entry.get("history", []) if r.get("revision_id") == revision), None)
    if rev_obj is None:
        raise ValueError(f"Revision {revision} not found for entry {entry_id}")
    path = rev_obj.get("build", {}).get("path")
    if not path:
        raise ValueError(f"No build path for entry {entry_id} rev {revision}")
    return Path(path)