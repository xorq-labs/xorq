from __future__ import annotations


HTTP_SCHEMES = ("http://", "https://")
CLOUD_SCHEMES = ("s3://", "gs://", "gcs://")
REMOTE_SCHEMES = HTTP_SCHEMES + CLOUD_SCHEMES

READ_IDENTITY_KEYS = frozenset({"mode", "schema", "temporary", "relocatable"})

READ_EXCLUDE_KEYS = frozenset({"hash_path", "read_path", "relocatable"})


# Backend names that identity/normalization dispatch keys on, kept here as the
# single source so a backend rename cannot leave a stale string behind in one
# dispatch table while the others move.  That has already happened: gh-1842
# shipped ``SnapshotStrategy.normalize_backend`` still holding the project's
# previous name ``"let"`` two renames after the fact, and the flight-name leak
# of gh-2229 is the same defect one axis over.  ``test_backend_names.py``
# asserts every name below is a live entry point in the ``xorq.backends``
# group, so the next rename fails in CI instead of silently under-keying.
DATAFUSION_BACKEND_NAMES = ("datafusion", "xorq_datafusion")
PANDAS_BACKEND_NAME = "pandas"
DUCKDB_BACKEND_NAME = "duckdb"
SQLITE_BACKEND_NAME = "sqlite"
BIGQUERY_BACKEND_NAME = "bigquery"

# Backends whose connection identity is fully determined by the backend name:
# no connection parameter distinguishes two instances for hashing purposes.
# Consumed by ``SnapshotStrategy.normalize_backend``.
NAME_ONLY_BACKEND_NAMES = (
    PANDAS_BACKEND_NAME,
    DUCKDB_BACKEND_NAME,
) + DATAFUSION_BACKEND_NAMES

# Every backend name the DatabaseTable dispatch chain in
# ``dasher/_relations.py`` special-cases before falling through to
# ``xorq_dasher``.  Derived, never re-spelled -- see REMOTE_SCHEMES above for
# the same subset-derivation pattern.
DISPATCHED_BACKEND_NAMES = DATAFUSION_BACKEND_NAMES + (
    PANDAS_BACKEND_NAME,
    DUCKDB_BACKEND_NAME,
    SQLITE_BACKEND_NAME,
    BIGQUERY_BACKEND_NAME,
)
