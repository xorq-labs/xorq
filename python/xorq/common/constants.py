from __future__ import annotations

from xorq.common.enums import BackendName


HTTP_SCHEMES = ("http://", "https://")
CLOUD_SCHEMES = ("s3://", "gs://", "gcs://")
REMOTE_SCHEMES = HTTP_SCHEMES + CLOUD_SCHEMES

# Read kwargs that do not bear identity. The path contributes content or stat
# metadata through the Read's normalize_method, read_path is location rather
# than identity (ADR-0006), table_name is gen_name'd per construction and
# memtables (``BundledSourceTypes.inmemory``) only marks a bundled memtable
# for its reload. Every other kwarg reaches the backend's read method and can
# change the rows it returns, so it is folded into the build hash and the
# cache key.
READ_NON_IDENTITY_KEYS = frozenset(
    {"hash_path", "read_path", "table_name", "memtables"}
)

READ_EXCLUDE_KEYS = frozenset({"hash_path", "read_path", "relocatable"})


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
