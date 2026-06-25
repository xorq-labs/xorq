from __future__ import annotations


HTTP_SCHEMES = ("http://", "https://")
CLOUD_SCHEMES = ("s3://", "gs://", "gcs://")
REMOTE_SCHEMES = HTTP_SCHEMES + CLOUD_SCHEMES

READ_IDENTITY_KEYS = frozenset({"mode", "schema", "temporary", "relocatable"})

READ_EXCLUDE_KEYS = frozenset({"hash_path", "read_path", "relocatable"})
