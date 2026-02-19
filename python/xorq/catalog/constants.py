METADATA_APPEND = ".metadata.yaml"
METADATA_INFIX = "metadata"
ENTRY_INFIX = "entries"
VALID_SUFFIXES = (".tar.gz", (PREFERRED_SUFFIX := ".tgz"))
CATALOG_YAML_NAME = "catalog.yaml"
REQUIRED_TGZ_NAMES = (
    (PROFILES_YAML_NAME := "profiles.yaml"),
    (METADATA_JSON_NAME := "metadata.json"),
    (EXPR_YAML_NAME := "expr.yaml"),
)
