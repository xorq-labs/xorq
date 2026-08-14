from __future__ import annotations

from xorq.common.enums import BackendName


HTTP_SCHEMES = ("http://", "https://")
CLOUD_SCHEMES = ("s3://", "gs://", "gcs://")
REMOTE_SCHEMES = HTTP_SCHEMES + CLOUD_SCHEMES

READ_IDENTITY_KEYS = frozenset({"mode", "schema", "temporary", "relocatable"})

READ_EXCLUDE_KEYS = frozenset({"hash_path", "read_path", "relocatable"})


# Dispatch sets derived from the ``BackendName`` enum — composed from members,
# never re-spelled as literals (gh-1842; see the enum's docstring).

# The two datafusion-flavored backends share every dispatch special case.
DATAFUSION_BACKEND_NAMES = (BackendName.DATAFUSION, BackendName.XORQ_DATAFUSION)

# Backends whose connection identity is fully determined by the backend name:
# no connection parameter distinguishes two instances for hashing purposes.
# Consumed by ``SnapshotStrategy.normalize_backend``.
NAME_ONLY_BACKEND_NAMES = frozenset(
    {
        BackendName.PANDAS,
        BackendName.DUCKDB,
        *DATAFUSION_BACKEND_NAMES,
    }
)

# Every backend name the DatabaseTable dispatch chain in
# ``dasher/_relations.py`` special-cases before falling through to
# ``xorq_dasher``.  Currently every member.
DISPATCHED_BACKEND_NAMES = frozenset(BackendName)
