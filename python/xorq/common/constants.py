from __future__ import annotations


REMOTE_SCHEMES = ("http://", "https://", "s3://", "gs://", "gcs://")

READ_IDENTITY_KEYS = frozenset({"mode", "schema", "temporary", "relocatable"})

READ_EXCLUDE_KEYS = ("hash_path", "read_path", "relocatable")
