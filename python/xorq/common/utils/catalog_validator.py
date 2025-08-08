import json
from pathlib import Path

import yaml
import jsonschema

def _load_schema() -> dict:
    schema_path = Path(__file__).parent.parent / 'catalog_schema_v0_2.json'
    return json.loads(schema_path.read_text())

__all__ = ["validate_catalog"]

def validate_catalog(path: str | Path) -> None:
    """
    Validate the catalog file at the given path against the v0.2 schema.
    Raises jsonschema.ValidationError on failure.
    """
    catalog_path = Path(path)
    data = yaml.safe_load(catalog_path.read_text())
    schema = _load_schema()
    jsonschema.validate(instance=data, schema=schema)